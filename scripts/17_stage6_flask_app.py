# 17_stage6_flask_app.py
# Этап 6: дашборд триажа алертов SIEM (одна страница, данные встроены в HTML).
#   Сверху: выбор типа конфигурации (ML / LLM / тандем) и карточка выбранной конфигурации,
#   затем сравнительные графики (F1 с доверительными интервалами, найдено и ложные тревоги, скорость, приоритеты),
#   затем разбор алертов с объяснениями LLM и отметкой верно/ошибка, таблица значений.
# Источники данных (запускать после 39_all_models_attack_metrics.py и 43/42):
#   results/stage_7/all_models_attack_metrics.csv  - метрики и 95 % ДИ всех конфигураций
#   results/stage_7/ml_predict_latency.json        - время ML на один алерт
#   results/stage_4/llm__*.csv, results/stage_7/{tandem_1,tandem_2,ml_only_*,ml_decides_llm_explains}.csv - решения по алертам
#   results/stage_2/experiment_sample.csv          - описания алертов и истинные метки
# Использование:
#   --static-only   записать results/stage_6/dashboard.html (открывается без сервера) и выйти
#   (без аргументов) запустить сервер http://0.0.0.0:5001
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS, ROOT

S2, S4, S6, S7 = (RESULTS / f"stage_{n}" for n in (2, 4, 6, 7))
TEMPLATE = ROOT / "src" / "dashboard_template.html"
OUT_HTML = S6 / "dashboard.html"

# id, название, тип (t - тандем, l - LLM, m - ML), ключ в all_models_attack_metrics.csv (префикс), файл решений, ключ в ml_predict_latency.json
CONFIGS = [
    # Главные тандемы: LightGBM решает и отбирает, LLM объясняет только отобранные алерты (файлы 47_tandem_explain.py)
    ("tm_q", "LightGBM → Qwen 3.5 4B", "t", "tandem_main__ollama__qwen3.5-4b-q4_K_M", S7 / "tandem_main__ollama__qwen3.5-4b-q4_K_M.csv", None),
    ("tm_gpt", "LightGBM → GPT-6-sol", "t", "tandem_main__polza__openai-gpt-6-sol", S7 / "tandem_main__polza__openai-gpt-6-sol.csv", None),
    ("tm_cl", "LightGBM → Claude Opus 5.5", "t", "tandem_main__claude__manual", S7 / "tandem_main__claude__manual.csv", None),
    ("cl", "Claude Opus 5.5", "l", "claude__claude-opus-5-5-manual__few__none", S4 / "llm__claude__claude-opus-5-5-manual__few__none.csv", None),
    ("gpt", "GPT-6-sol (облако)", "l", "polza__openai-gpt-6-sol__few__none", S4 / "llm__polza__openai-gpt-6-sol__few__none.csv", None),
    ("l31", "llama3.1 8B", "l", "ollama__llama3.1-latest__few__none", S4 / "llm__ollama__llama3.1-latest__few__none.csv", None),
    ("q", "Qwen 3.5 4B", "l", "ollama__qwen3.5-4b-q4_K_M__few__none", S4 / "llm__ollama__qwen3.5-4b-q4_K_M__few__none.csv", None),
    ("qf", "Qwen 3.5 4B + частоты", "l", "ollama__qwen3.5-4b-q4_K_M__few__freq", S4 / "llm__ollama__qwen3.5-4b-q4_K_M__few__freq.csv", None),
    ("g2", "gemma2 2B", "l", "ollama__gemma2-2b__few__none", S4 / "llm__ollama__gemma2-2b__few__none.csv", None),
    ("l32", "llama3.2 3B", "l", "ollama__llama3.2-3b__few__none", S4 / "llm__ollama__llama3.2-3b__few__none.csv", None),
    ("lgb", "LightGBM", "m", "LightGBM θ", S7 / "ml_only_lightgbm.csv", "LightGBM"),
    ("xgb", "XGBoost", "m", "XGBoost θ", S7 / "ml_only_xgboost.csv", "XGBoost"),
    ("lr", "Logistic Regression", "m", "LogisticRegression θ", S7 / "ml_only_logreg.csv", "LogisticRegression"),
    ("rf", "Random Forest", "m", "RandomForest θ", S7 / "ml_only_randomforest.csv", "RandomForest"),
    ("svm", "LinearSVM", "m", "LinearSVM θ", S7 / "ml_only_linearsvm.csv", "LinearSVM"),
    ("lgbt", "LightGBM, только текст", "m", "LightGBM[text] θ", None, "LightGBM[text]"),
    # Эксперимент: LLM как второе мнение (алерт уходит аналитику, если его отметил ML ИЛИ LLM)
    ("tandem_2", "ИЛИ: LightGBM или GPT-6-sol", "x", "tandem_2", S7 / "tandem_2.csv", None),
    ("tandem_1", "ИЛИ: LightGBM или Qwen + частоты", "x", "tandem_1", S7 / "tandem_1.csv", None),
]
CLOUD = {"tm_gpt", "tm_cl", "cl", "gpt", "tandem_2"}  # данные уходят во внешний облачный сервис; остальное считается локально
PRI = ("low", "medium", "high", "critical")


def _s(v):
    return "" if pd.isna(v) else str(v).strip()


# Объяснения Claude получены вручную, а время и цена измерены через API Polza на 5 алертах (скрипт 55).
CL_M = json.load(open(S7 / "claude_polza_measure.json", encoding="utf-8"))
LAT_MEAS = {"tm_cl": (CL_M["lat_s"], f"замер на {CL_M['n_trial']} алертах через API")}
LAT_EST = {"cl": (CL_M["lat_s"], "оценка по замеру тандема")}  # «просто LLM» отдельно по API не мерили
COST_TXT = {"tm_q": "0 ₽", "tm_gpt": "≈ 48 ₽", "tm_cl": f"≈ {CL_M['cost_102']} ₽"}  # за 102 объяснения
QUAL_MODEL = {"tm_q": "Qwen 3.5 4B", "tm_gpt": "GPT-6-sol", "tm_cl": "Claude"}


def quality_scores():
    """Оценка качества объяснений (слепая выборка 30 алертов, два оценщика, шкала 0-2): {модель: [общая, что, почему, рекомендация]}."""
    import re
    import openpyxl
    root = Path(__file__).resolve().parents[1]
    xl, jd = root / "reports" / "Оценка_объяснений_слепая.xlsx", root / "reports" / "judge"
    if not xl.exists():
        return {}
    key = {}
    for r in openpyxl.load_workbook(xl)["Оценка"].iter_rows(min_row=4, values_only=True):
        if r[1] in ("A", "B", "C") and isinstance(r[0], (int, float)):
            key[(int(r[0]), r[1])] = r[7].split(" (")[0]
    acc = {}
    for name in ("deepseek", "grok"):
        f = jd / f"answers_{name}_part_1.txt"
        if not f.exists():
            continue
        for n, v, a, b, c in re.findall(r"(\d+)\s+([ABC])\s+([012])\s+([012])\s+([012])", f.read_text(encoding="utf-8")):
            acc.setdefault(key.get((int(n), v)), []).append([int(a), int(b), int(c)])
    out = {}
    for m, rows in acc.items():
        if m is None:
            continue
        cols = [sum(r[i] for r in rows) / len(rows) for i in range(3)]
        out[m] = [round(sum(cols) / 3, 2)] + [round(x, 2) for x in cols]
    return out


def build_data():
    metrics = pd.read_csv(S7 / "all_models_attack_metrics.csv")
    lat = json.load(open(S7 / "ml_predict_latency.json"))
    sample = pd.read_csv(S2 / "experiment_sample.csv")
    base = metrics[metrics["family"] == "Базовая линия"].iloc[0]
    cfg, queues = [], {}
    qual = quality_scores()
    for cid, name, fam, mkey, path, lkey in CONFIGS:
        hit = metrics[metrics["model"].str.startswith(mkey)]
        if hit.empty:
            print("пропуск (нет метрик):", name)
            continue
        row = hit.iloc[0]
        if path is not None and not Path(path).exists():
            print("пропуск (нет файла):", path)
            continue
        pri = None
        if path is not None:
            d = pd.read_csv(path)
            if len(d) != len(sample):
                print("пропуск (неполный файл):", path)
                continue
            has_p = "ml_proba" in d
            queues[cid] = [[r.priority, round(float(r.ml_proba), 3) if has_p and pd.notna(r.ml_proba) else None,
                            _s(getattr(r, "source", "")), _s(r.what_happened), _s(r.why_suspicious), _s(r.recommendation)]
                           for r in d.itertuples()]
            pri = [int((d["priority"] == p).sum()) for p in PRI]
        if lkey and lkey in lat:
            latency = float(lat[lkey])
        else:
            latency = None if pd.isna(row["latency_s"]) else float(row["latency_s"])  # None — не измерялось
        est = cid in LAT_EST
        lnote = None
        if cid in LAT_MEAS:
            latency, lnote = LAT_MEAS[cid]
        elif est:
            latency, lnote = LAT_EST[cid]
        cfg.append(dict(id=cid, name=name, fam=fam, found=int(row["found"]), fp=int(row["fp"]), recall=round(float(row["recall"]), 3),
                        prec=round(float(row["prec"]), 3), f1=round(float(row["F1"]), 3), lo=round(float(row["F1_lo"]), 3),
                        hi=round(float(row["F1_hi"]), 3), lat=latency, pri=pri, loc="c" if cid in CLOUD else "l", q=qual.get(QUAL_MODEL.get(cid)), est=est, lnote=lnote, cost=COST_TXT.get(cid)))
    alerts = [[r.description, r.label, r.group if "group" in sample else "", r.scenario] for r in sample.itertuples()]
    n_att = int(sample["label"].ne("benign").sum())
    return dict(cfg=cfg, base=dict(f1=round(float(base["F1"]), 3), lo=round(float(base["F1_lo"]), 3), hi=round(float(base["F1_hi"]), 3),
                                   found=n_att, fp=len(sample) - n_att),
                att=n_att, ben=len(sample) - n_att, alerts=alerts, Q=queues)


def build_html():
    d = build_data()
    html = TEMPLATE.read_text(encoding="utf-8")
    for k, v in (("__ATT__", d["att"]), ("__BEN__", d["ben"]), ("__N__", d["att"] + d["ben"])):
        html = html.replace(k, str(v))
    payload = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return html.replace("__DATA__", payload)


def main():
    S6.mkdir(parents=True, exist_ok=True)
    if "--static-only" in sys.argv:
        for old in glob.glob(str(S6 / "dashboard_*.html")):  # устаревшие страницы прежней версии
            os.remove(old)
        html = build_html()
        OUT_HTML.write_text(html, encoding="utf-8")
        print("written:", OUT_HTML, len(html) // 1024, "KB")
        return
    from flask import Flask
    app = Flask(__name__)

    @app.route("/")
    def index():
        return build_html()

    import socket
    lan = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")]
    print("Launching at http://127.0.0.1:5001 — open in browser.")
    print("Доступно по сети: " + "; ".join("http://" + ip + ":5001" for ip in lan))
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main()
