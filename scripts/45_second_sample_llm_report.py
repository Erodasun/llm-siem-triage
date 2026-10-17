# 45_second_sample_llm_report.py
# Итог по второй выборке: ML (скрипт 44) + локальные LLM (14 --sample2) + Тандем 1 (LightGBM ИЛИ Qwen+freq, приоритет >= high).
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import attack_metrics, is_positive, trivial_baseline
from src.config import LEVEL, stage_dir, utf8_console


def boot(y, pred, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    f = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        f.append(attack_metrics(y[i], pred[i])["F1"])
    return float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))


def main():
    utf8_console()
    S8 = stage_dir(8)
    sp = pd.read_csv(S8 / "experiment_sample_2.csv")
    y = sp["label"].ne("benign").values
    rows = []
    ml = pd.read_csv(S8 / "ml_sample2.csv").set_index("model")
    z = np.load(S8 / "ml_probas_sample2.npz")
    for k, r in ml.iterrows():
        pred = z[k] >= r["theta"]
        rows.append(("ML", k, pred))
    llm = {}
    for f in sorted(S8.glob("llm2__*.csv")):
        d = pd.read_csv(f)
        if len(d) != len(sp):
            print("пропуск (не завершён):", f.name, len(d))
            continue
        name = f.stem.replace("llm2__", "")
        llm[name] = d
        rows.append(("LLM", name, is_positive(d["priority"])))
    th = ml.loc["LightGBM", "theta"]
    q = next((k for k in llm if "qwen" in k and k.endswith("freq")), None)
    if q:
        lvl = llm[q]["priority"].map(LEVEL).fillna(-1).values
        rows.append(("Тандем", "tandem_1 (LightGBM или Qwen+freq≥high)", (z["LightGBM"] >= th) | (lvl >= LEVEL["high"])))
    rows.append(("База", "всё — атака", np.ones(len(y), bool)))
    out = []
    for fam, nm, pred in rows:
        m = attack_metrics(y, pred)
        lo, hi = boot(y, np.asarray(pred))
        out.append({"family": fam, "model": nm, "found": m["found"], "fp": m["fp"], "recall": m["recall"], "prec": m["prec"],
                    "F1": m["F1"], "F1_lo": lo, "F1_hi": hi})
    res = pd.DataFrame(out)
    res.to_csv(S8 / "all_models_sample2.csv", index=False, encoding="utf-8-sig")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
