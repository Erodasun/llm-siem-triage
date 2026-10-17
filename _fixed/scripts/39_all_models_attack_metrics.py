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

S4, S8 = stage_dir(4), stage_dir(8)


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
    rows = [{"family": "Базовая линия", "model": "Всё — атака", **trivial_baseline(y), **group_cols(group, np.ones(len(y), bool)),
             "errors": 0, "latency_s": 0.0}]
    for f in sorted(S4.glob("llm__*.csv")):
        d = pd.read_csv(f).sort_values("i")
        if len(d) != len(y):
            print(f"пропуск (строк {len(d)}, в выборке {len(y)}):", f.name)
            continue
        pred = is_positive(d["priority"])
        rows.append({"family": "LLM", "model": f.stem.replace("llm__", ""), **attack_metrics(y, pred), **group_cols(group, pred),
                     "errors": int((d["priority"] == "error").sum()), "latency_s": round(d["latency_s"].mean(), 3)})
    if (S8 / "ml_probas.npz").exists():
        probas = np.load(S8 / "ml_probas.npz")
        th = json.load(open(S8 / "ml_thresholds.json"))
        lat = json.load(open(S8 / "ml_predict_latency.json"))
        for k in probas.files:
            pred = probas[k] >= th[k]
            rows.append({"family": "ML", "model": f"{k} θ={th[k]}", **attack_metrics(y, pred), **group_cols(group, pred),
                         "errors": 0, "latency_s": lat.get(k, np.nan)})
    for f in sorted(S8.glob("tandem_*.csv")):
        if f.stem.startswith("tandem_sweep"):
            continue
        d = pd.read_csv(f).sort_values("i")
        pred = is_positive(d["priority"])
        rows.append({"family": "Тандем", "model": f.stem, **attack_metrics(y, pred), **group_cols(group, pred),
                     "errors": 0, "latency_s": round(d["latency_s"].mean(), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(S8 / "all_models_attack_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    for fam, g in out.groupby("family", sort=False):
        print(f"\n=== {fam} ===")
        print(g.drop(columns="family").sort_values("F1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
