# 39_all_models_attack_metrics.py
# Единая таблица по классу «атака» для всех моделей на оценочной выборке: LLM (results/stage_4/llm__*.csv),
# ML (порог θ с временной валидации), тандемы и тривиальная модель «всё — атака».
# Для выборки с колонкой group дополнительно: recall_easy, recall_hard, fp_hard_benign (доля фона среди парных «трудных»).
# Атакой считается priority ∈ {medium, high, critical}; ответы с ошибкой (priority=error) считаются пропуском.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import attack_metrics, is_positive, trivial_baseline
from src.config import SAMPLE, stage_dir, utf8_console

S4, S7 = stage_dir(4), stage_dir(7)


def boot_ci(y, pred, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    f = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        f.append(attack_metrics(y[i], pred[i])["F1"])
    lo, hi = np.percentile(f, [2.5, 97.5])
    return {"F1_lo": round(float(lo), 3), "F1_hi": round(float(hi), 3)}


def group_cols(group, pred):
    out = {}
    if group is None:
        return out
    for g, name in (("easy_attack", "recall_easy"), ("hard_attack", "recall_hard")):
        m = group == g
        out[name] = round(float(pred[m].mean()), 3) if m.any() else np.nan
    m = group == "hard_benign"
    out["fp_hard_benign"] = round(float(pred[m].mean()), 3) if m.any() else np.nan
    return out


def main():
    utf8_console()
    sample = pd.read_csv(SAMPLE)
    y = sample["label"].ne("benign").values
    group = sample["group"].values if "group" in sample else None
    rows = [{"family": "Базовая линия", "model": "Всё — атака", **trivial_baseline(y), **boot_ci(y, np.ones(len(y), bool)), **group_cols(group, np.ones(len(y), bool)),
             "errors": 0, "latency_s": 0.0}]
    for f in sorted(S4.glob("llm__*.csv")):
        d = pd.read_csv(f).sort_values("i")
        if len(d) != len(y):
            print(f"пропуск (строк {len(d)}, в выборке {len(y)}):", f.name)
            continue
        pred = is_positive(d["priority"])
        rows.append({"family": "LLM", "model": f.stem.replace("llm__", ""), **attack_metrics(y, pred), **boot_ci(y, pred), **group_cols(group, pred),
                     "errors": int((d["priority"] == "error").sum()), "latency_s": round(d["latency_s"].mean(), 3)})
    if (S7 / "ml_probas.npz").exists():
        probas = np.load(S7 / "ml_probas.npz")
        th = json.load(open(S7 / "ml_thresholds.json"))
        lat = json.load(open(S7 / "ml_predict_latency.json"))
        for k in probas.files:
            pred = probas[k] >= th[k]
            rows.append({"family": "ML", "model": f"{k} θ={th[k]}", **attack_metrics(y, pred), **boot_ci(y, pred), **group_cols(group, pred),
                         "errors": 0, "latency_s": lat.get(k, np.nan)})
    # Главные тандемы: tandem_main__<бэкенд>__<модель>.csv (ML отбирает, LLM объясняет; решает ML).
    # Эксперимент «LLM как второе мнение»: tandem_1.csv, tandem_2.csv (алерт уходит аналитику, если отметил ML ИЛИ LLM).
    for f in sorted(S7.glob("tandem_*.csv")):
        if "anthropic-claude" in f.stem:  # проба на 5 алертах: только замер скорости и цены (скрипт 55)
            continue
        if not (f.stem.startswith("tandem_main__") or f.stem in ("tandem_1", "tandem_2")):
            continue
        d = pd.read_csv(f).sort_values("i")
        if len(d) != len(y):
            print(f"пропуск (строк {len(d)}, в выборке {len(y)}):", f.name)
            continue
        pred = is_positive(d["priority"])
        main = f.stem.startswith("tandem_main__")
        if main:
            llm = d[d["source"] == "ML+LLM"]
            manual = llm["llm_model"].astype(str).str.contains("вручную").any()
            lat = np.nan if manual else round(llm["latency_s"].mean(), 3)  # время на один отобранный алерт (ML + LLM)
        else:
            lat = round(d["latency_s"].mean(), 3)
        rows.append({"family": "Тандем" if main else "Эксперимент", "model": f.stem, **attack_metrics(y, pred), **boot_ci(y, pred),
                     **group_cols(group, pred), "errors": 0, "latency_s": lat})
    out = pd.DataFrame(rows)
    out.to_csv(S7 / "all_models_attack_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    for fam, g in out.groupby("family", sort=False):
        print(f"\n=== {fam} ===")
        print(g.drop(columns="family").sort_values("F1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
