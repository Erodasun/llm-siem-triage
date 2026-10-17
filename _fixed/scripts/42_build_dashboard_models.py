# 42_build_dashboard_models.py
# Готовит per-alert CSV для дашборда из результатов 43_build_tandem_study.py:
#   ml_only_<model>.csv          - чистый ML (порог θ с валидации, уровни по квантилям вероятности, без объяснений);
#   ml_decides_llm_explains.csv  - решает LightGBM, LLM объясняет только эскалированные алерты.
# Тандемы 1 и 2 собирает сам 43_build_tandem_study.py (results/stage_8/tandem_1.csv, tandem_2.csv).
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FIELDS, attack_metrics, is_positive, levels_from_score
from src.config import SAMPLE, stage_dir, utf8_console

S4, S8 = stage_dir(4), stage_dir(8)
ML_FILES = {"XGBoost": "ml_only_xgboost", "LightGBM": "ml_only_lightgbm", "RandomForest": "ml_only_randomforest",
            "LogisticRegression": "ml_only_logreg", "LinearSVM": "ml_only_linearsvm"}


def report(name, d):
    m = attack_metrics(d["label"].ne("benign"), is_positive(d["priority"]))
    print(f"{name:<28} {m} {d['priority'].value_counts().to_dict()}")


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--explainer", default="llm__ollama__qwen3.5-4b-q4_K_M__few__freq")
    a = ap.parse_args()
    sample = pd.read_csv(SAMPLE)
    probas = np.load(S8 / "ml_probas.npz")
    th = json.load(open(S8 / "ml_thresholds.json"))
    lat = json.load(open(S8 / "ml_predict_latency.json"))
    for model, fname in ML_FILES.items():
        p = probas[model]
        d = pd.DataFrame({"i": range(len(sample)), "label": sample["label"],
                          "priority": levels_from_score(p, p >= th[model]), "ml_proba": np.round(p, 3),
                          "latency_s": lat[model]})
        for f in FIELDS:
            d[f] = ""
        d.to_csv(S8 / f"{fname}.csv", index=False, encoding="utf-8-sig")
        report(fname, d)
    exp_path = S4 / f"{a.explainer}.csv"
    if not exp_path.exists():
        print("нет файла объяснений:", exp_path)
        return
    llm = pd.read_csv(exp_path).sort_values("i").reset_index(drop=True)
    p = probas["LightGBM"]
    flag = p >= th["LightGBM"]
    d = pd.DataFrame({"i": llm["i"], "label": llm["label"], "priority": levels_from_score(p, flag),
                      "ml_proba": np.round(p, 3), "source": np.where(flag, "ML", ""),
                      "latency_s": np.round(lat["LightGBM"] + np.where(flag, llm["latency_s"], 0.0), 6)})
    for f in FIELDS:
        d[f] = np.where(flag, llm[f].fillna(""), "")
    d.to_csv(S8 / "ml_decides_llm_explains.csv", index=False, encoding="utf-8-sig")
    report("ml_decides_llm_explains", d)


if __name__ == "__main__":
    main()
