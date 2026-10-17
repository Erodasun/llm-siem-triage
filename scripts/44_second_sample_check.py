# 44_second_sample_check.py
# Проверка устойчивости результатов: вторая независимая оценочная выборка (другое зерно, без пересечения с первой).
# Модели и пороги θ те же (θ выбран на валидации из train, на второй выборке ничего не подбирается).
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import FULL, SAMPLE, TEST, TEST_ATTACKS, TRAIN, stage_dir, utf8_console
from src.common import attack_metrics, trivial_baseline

SEED2 = 7


def load43():
    spec = importlib.util.spec_from_file_location("s43", Path(__file__).resolve().parent / "43_build_tandem_study.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_sample2(size=200, tau=0.5, hard_max=0.95, easy_per_type=12, hard_per_type=8):
    full = pd.read_csv(FULL, usecols=["description", "label"])
    share = full.groupby("description")["label"].agg(lambda s: (s == "benign").mean())
    del full
    df = pd.read_csv(TEST)
    df.insert(0, "alert_id", range(len(df)))
    used = set(pd.read_csv(SAMPLE)["alert_id"])
    df["benign_share"] = df["description"].map(share)
    df = df[~df["alert_id"].isin(used)]
    window = df[df["label"].isin(TEST_ATTACKS)]
    easy = window[window["benign_share"] < tau]
    hard = window[(window["benign_share"] >= tau) & (window["benign_share"] < hard_max)]
    tk = lambda d, n: pd.concat([g.sample(min(n, len(g)), random_state=SEED2) for _, g in d.groupby("label")])
    ea, ha = tk(easy, easy_per_type).assign(group="easy_attack"), tk(hard, hard_per_type).assign(group="hard_attack")
    pool = df[df["label"] == "benign"]
    paired = [pool[pool["description"] == d].sample(min(n, (pool["description"] == d).sum()), random_state=SEED2)
              for d, n in ha["description"].value_counts().items()]
    hb = pd.concat(paired).assign(group="hard_benign")
    rest = pool.drop(hb.index)
    eb = rest.sample(max(size - len(ea) - len(ha) - len(hb), 0), random_state=SEED2).assign(group="easy_benign")
    s = pd.concat([ea, ha, hb, eb]).sample(frac=1, random_state=SEED2).reset_index(drop=True)
    return s


def main():
    utf8_console()
    m = load43()
    out = stage_dir(8)
    s2 = build_sample2()
    s2.to_csv(out / "experiment_sample_2.csv", index=False)
    print("выборка 2:", len(s2), "атак", int(s2["label"].ne("benign").sum()), s2["group"].value_counts().to_dict(), flush=True)
    y = s2["label"].ne("benign").values
    full = pd.read_csv(FULL, usecols=["scenario", "time", "host", "description"])
    tr = pd.read_csv(TRAIN, usecols=m.COLS)
    tr = pd.concat([g.sample(min(m.PER_LABEL, len(g)), random_state=m.SEED) for _, g in tr.groupby("label")])
    fit, val = m.temporal_split(tr)
    pos = val["label"].ne("benign")
    k = int(min(pos.sum(), (~pos).sum()))
    val = pd.concat([val[pos].sample(k, random_state=m.SEED), val[~pos].sample(k, random_state=m.SEED)])
    feats = m.Features(full, fit)
    y_fit, y_val = fit["label"].ne("benign").values, val["label"].ne("benign").values
    rows, probas = [], {}
    cache = {}
    plan = [(n, ("bursts", "context")) for n in m.make_models()] + [("LightGBM", ())]
    for name, groups in plan:
        key = name if groups == ("bursts", "context") else f"{name}[text]"
        if groups not in cache:
            cache[groups] = tuple(feats.matrix(d, groups) for d in (fit, val, s2))
        Xf, Xv, Xs = cache[groups]
        mod = m.make_models()[name]
        mod.fit(Xf, y_fit)
        pv, ps = m.score(mod, Xv), m.score(mod, Xs)
        th = m.best_theta(y_val, pv)
        r = {"model": key, "theta": th, **attack_metrics(y, ps >= th)}
        rows.append(r)
        probas[key] = ps
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(out / "ml_sample2.csv", index=False, encoding="utf-8-sig")
    np.savez(out / "ml_probas_sample2.npz", **probas)
    json.dump(trivial_baseline(y), open(out / "baseline_sample2.json", "w"), default=float)
    print("baseline:", trivial_baseline(y))


if __name__ == "__main__":
    main()
