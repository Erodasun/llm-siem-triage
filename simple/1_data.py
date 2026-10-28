# 1_data.py — ШАГ 1. Данные: собираем алерты, ставим метки, делим по времени и на группы сетей.
#
# Что происходит:
#   1) Из 16 файлов AIT-ADS (Wazuh и AMiner по 8 сетям) берём одинаковые поля каждого алерта.
#      Пропускаем записи AMiner в режиме самообучения и записи без времени.
#   2) Метка: если время алерта попадает в интервал атаки из labels.csv — название атаки, иначе benign (фон).
#   3) Каждая сеть делится по времени: до первого алерта веб-оболочки (webshell) — ранняя часть (early),
#      после — поздняя (late). Поздняя часть — это тест.
#   4) Сети делятся на две группы, A и B: модель, которую проверяют на группе A, не видит её позднюю часть.
# Результат: simple/results/alerts.parquet — одна таблица, строка = алерт, rid = номер строки.
#
# Запуск: python simple/1_data.py            (если размеченная таблица data/processed/02_labeled_alerts.csv уже есть,
#                                              она используется; --rebuild — собрать заново из исходных json)
import csv, json, sys
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from src.config import GROUPS, LABELED, LABELS_CSV, LOG_CHARS, OUT, RAW, SCENARIOS, utf8

COLS = ["scenario", "source", "time", "host", "description", "full_log", "label"]


# --- разбор исходных файлов (нужен, только если размеченной таблицы ещё нет) --------------------
def parse_raw():
    attacks = {}
    for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")):
        attacks.setdefault(r["scenario"], []).append((r["attack"], float(r["start"]), float(r["end"])))

    def when(o):
        ld = o.get("LogData", {})
        if ld.get("DetectionTimestamp"):
            return ld["DetectionTimestamp"][0]
        if ld.get("Timestamps"):
            return ld["Timestamps"][0]
        ts = o.get("@timestamp")
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else None
        except ValueError:
            return None

    LABELED.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELED, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for sc in SCENARIOS:
            for src in ("aminer", "wazuh"):
                for line in open(RAW / f"{sc}_{src}.json", encoding="utf-8"):
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ac = o.get("AnalysisComponent", {})
                    if isinstance(ac, dict) and ac.get("TrainingMode") is True:      # AMiner ещё учится
                        continue
                    t = when(o)
                    if t is None:
                        continue
                    if "LogData" in o:                                                 # AMiner
                        raw = o["LogData"].get("RawLogData", "")
                        raw = " ".join(raw) if isinstance(raw, list) else raw
                        row = [sc, "aminer", t, "", ac.get("AnalysisComponentName", "") or "", raw]
                    else:                                                              # Wazuh (и Suricata через Wazuh)
                        row = [sc, "wazuh", t, o.get("agent", {}).get("name", "") or "",
                               o.get("rule", {}).get("description", "") or "", o.get("full_log", "")]
                    label = next((a for a, s, e in attacks[sc] if s <= t <= e), "benign")
                    w.writerow(row + [label])
            print("прочитан сценарий", sc, flush=True)


def main():
    utf8()
    if "--rebuild" in sys.argv or not LABELED.exists():
        parse_raw()
    OUT.mkdir(parents=True, exist_ok=True)

    # граница каждой сети: время первого алерта веб-оболочки
    meta = pd.read_csv(LABELED, usecols=["scenario", "time", "label"])
    cut = meta[meta.label == "webshell"].groupby("scenario")["time"].min().to_dict()
    print("граница ранней и поздней части:", {k: pd.Timestamp(v, unit="s").strftime("%d.%m %H:%M") for k, v in cut.items()})
    del meta
    group_of = {s: g for g, l in GROUPS.items() for s in l}

    writer, rid = None, 0
    for ch in pd.read_csv(LABELED, usecols=COLS, chunksize=300_000):
        ch.insert(0, "rid", range(rid, rid + len(ch)))
        rid += len(ch)
        ch["host"] = ch["host"].fillna("").astype(str)
        ch["description"] = ch["description"].fillna("").astype(str)
        ch["full_log"] = ch["full_log"].fillna("").astype(str).str[:LOG_CHARS]
        ch["part"] = (ch["time"] >= ch["scenario"].map(cut)).map({True: "late", False: "early"})
        ch["group"] = ch["scenario"].map(group_of)
        t = pa.Table.from_pandas(ch, preserve_index=False)
        writer = writer or pq.ParquetWriter(OUT / "alerts.parquet", t.schema)
        writer.write_table(t)
    writer.close()

    a = pd.read_parquet(OUT / "alerts.parquet", columns=["part", "label"])
    tab = pd.crosstab(a["label"], a["part"])
    print(f"\nвсего алертов: {len(a):,}".replace(",", " "))
    print(tab.to_string())
    tab.to_csv(OUT / "labels_by_part.csv")


if __name__ == "__main__":
    main()
