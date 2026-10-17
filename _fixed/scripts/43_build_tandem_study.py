# 43_build_tandem_study.py
# Этап 8 (заменяет 32-38, 40, 41 и прежний 43).
#   A) пять ML-моделей на одном «богатом» наборе признаков + абляция групп признаков для LightGBM;
#      порог θ выбирается на временной валидации из train (последние 20 % каждого сценария),
#      оценка — на оценочной выборке и на полном тесте; θ, подобранный прямо на выборке, выводится отдельно как «oracle»;
#   B) перебор пар «ML × LLM» по всем результатам results/stage_4/llm__*.csv (разведочный, на оценочной выборке);
#   C) сборка итоговых тандемов для дашборда.
# Запуск: python scripts\43_build_tandem_study.py [--tandem1 <файл LLM>] [--tandem2 <файл LLM>] [--skip-full-test]
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (FIELDS, attack_metrics, burst_counts, is_positive, level_of, levels_from_score,
                        time_index, trivial_baseline)
from src.config import FULL, SAMPLE, SEED, TEST, TRAIN, stage_dir, utf8_console

WINDOWS = (10, 30, 60, 300, 900, 3600)
THETAS = np.round(np.arange(0.05, 0.96, 0.05), 2)
PER_LABEL = 150000 // 11
S4, S8 = stage_dir(4), stage_dir(8)
COLS = ["scenario", "source", "time", "host", "description", "full_log", "label"]


def text_of(df):
    return df["description"].fillna("") + " " + df["full_log"].fillna("").astype(str).str[:400]


class Features:
    def __init__(self, full, train):
        full = full.assign(host=full["host"].fillna(""))
        self.idx_sd = time_index(full, ("scenario", "description"))
        self.idx_hd = time_index(full, ("host", "description"))
        self.host_freq = train["host"].fillna("").value_counts()
        self.word = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True)
        self.ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        self.cache = {}
        txt = text_of(train)
        self.word.fit(txt)
        self.char.fit(txt)
        self.ohe.fit(train[["scenario", "source"]].astype(str))

    def numeric(self, df):
        host = df["host"].fillna("")
        sd, hd, t = list(zip(df["scenario"], df["description"])), list(zip(host, df["description"])), df["time"].values
        f = {}
        for w in WINDOWS:
            f[f"bsd_{w}"] = burst_counts(self.idx_sd, sd, t, w)
            f[f"bhd_{w}"] = burst_counts(self.idx_hd, hd, t, w)
        f = pd.DataFrame(f)
        f["burst_ratio_60"] = f["bsd_60"] / (f["bsd_3600"] / 60.0 + 1.0)
        f["burst_ratio_300"] = f["bsd_300"] / (f["bsd_3600"] / 12.0 + 1.0)
        for w in WINDOWS:
            f[f"bsd_{w}"] = np.log1p(f[f"bsd_{w}"])
            f[f"bhd_{w}"] = np.log1p(f[f"bhd_{w}"])
        ctx = pd.DataFrame({"host_freq": host.map(self.host_freq).fillna(0).values,
                            "hour": (t // 3600 % 24).astype(int), "day": ((t // 86400).astype(int) % 7)})
        return f, ctx

    def matrix(self, df, groups):
        key = id(df)
        if key not in self.cache:
            txt = text_of(df)
            self.cache[key] = (self.word.transform(txt), self.char.transform(txt), *self.numeric(df))
        word, char, bursts, ctx = self.cache[key]
        blocks = [word, char]
        if "bursts" in groups:
            blocks.append(csr_matrix(bursts.values.astype(np.float32)))
        if "context" in groups:
            blocks.append(self.ohe.transform(df[["scenario", "source"]].astype(str)))
            blocks.append(csr_matrix(ctx.values.astype(np.float32)))
        return hstack(blocks).tocsr()


def make_models():
    return {
        "XGBoost": XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                                 tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=500, max_depth=7, learning_rate=0.08, subsample=0.9, subsample_freq=1,
                                   colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=SEED, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0),
        "LinearSVM": LinearSVC(max_iter=3000, random_state=SEED),
    }


def score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


def best_theta(y, p):
    rows = [(th, attack_metrics(y, p >= th)["F1"]) for th in THETAS]
    return max(rows, key=lambda r: r[1])[0]


def temporal_split(tr):
    tr = tr.sort_values("time")
    cut = tr.groupby("scenario")["time"].transform(lambda s: s.quantile(0.8))
    return tr[tr["time"] < cut], tr[tr["time"] >= cut]


def part_a(skip_full_test):
    full = pd.read_csv(FULL, usecols=["scenario", "time", "host", "description"])
    tr = pd.read_csv(TRAIN, usecols=COLS)
    tr = pd.concat([g.sample(min(PER_LABEL, len(g)), random_state=SEED) for _, g in tr.groupby("label")])
    fit, val = temporal_split(tr)
    sp = pd.read_csv(SAMPLE)
    feats = Features(full, fit)
    y_fit, y_val, y_sp = (d["label"].ne("benign").values for d in (fit, val, sp))
    te = None if skip_full_test else pd.read_csv(TEST, usecols=COLS)
    plan = [(m, ("bursts", "context")) for m in make_models()] + [("LightGBM", ()), ("LightGBM", ("bursts",))]
    rows, probas, latency = [], {}, {}
    cache = {}
    for name, groups in plan:
        key = name if groups == ("bursts", "context") else f"{name}[{'+'.join(('text',) + groups)}]"
        if groups not in cache:
            cache[groups] = tuple(feats.matrix(d, groups) for d in (fit, val, sp)) + (
                (feats.matrix(te, groups),) if te is not None else (None,))
        X_fit, X_val, X_sp, X_te = cache[groups]
        model = make_models()[name]
        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_s = time.perf_counter() - t0
        p_val, p_sp = score(model, X_val), score(model, X_sp)
        dt = []
        for _ in range(20):
            t1 = time.perf_counter()
            score(model, X_sp)
            dt.append(time.perf_counter() - t1)
        latency[key] = float(np.median(dt) / X_sp.shape[0])
        th = best_theta(y_val, p_val)
        th_oracle = best_theta(y_sp, p_sp)
        row = {"model": key, "fit_s": round(fit_s, 1), "theta_val": th,
               **{f"{k}": v for k, v in attack_metrics(y_sp, p_sp >= th).items()},
               "theta_oracle": th_oracle, "F1_oracle": attack_metrics(y_sp, p_sp >= th_oracle)["F1"]}
        if X_te is not None:
            mt = attack_metrics(te["label"].ne("benign").values, score(model, X_te) >= th)
            row.update({"test_recall": mt["recall"], "test_prec": mt["prec"], "test_F1": mt["F1"]})
        if name == "LightGBM" and groups == ("bursts", "context"):
            names = (list(feats.word.get_feature_names_out()) + list(feats.char.get_feature_names_out())
                     + [f"num:{c}" for c in feats.numeric(sp.head(1))[0].columns]
                     + [f"cat:{c}" for c in feats.ohe.get_feature_names_out()] + ["ctx:host_freq", "ctx:hour", "ctx:day"])
            imp = pd.Series(model.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)
            imp.head(30).to_csv(S8 / "lightgbm_feature_importance.csv", encoding="utf-8-sig")
            print("top-10 gain:", imp.head(10).round(0).to_dict())
        rows.append(row)
        probas[key] = p_sp
        print(row, flush=True)
    res = pd.DataFrame(rows)
    base = trivial_baseline(y_sp)
    res.to_csv(S8 / "ml_models_comparison.csv", index=False, encoding="utf-8-sig")
    np.savez(S8 / "ml_probas.npz", **probas)
    json.dump(latency, open(S8 / "ml_predict_latency.json", "w"), indent=1)
    json.dump({k: float(r["theta_val"]) for k, r in res.set_index("model").iterrows()},
              open(S8 / "ml_thresholds.json", "w"), indent=1)
    print("\nтривиальный «всё — атака» на выборке:", base)
    print(res.to_string(index=False))


def llm_results():
    n = len(pd.read_csv(SAMPLE, usecols=["label"]))
    out = {}
    for f in sorted(S4.glob("llm__*.csv")):
        d = pd.read_csv(f)
        if len(d) == n:
            out[f.stem] = d.sort_values("i").reset_index(drop=True)
    return out


def part_b(ml, llms, y):
    rows = []
    for mname, p in ml.items():
        for lname, d in llms.items():
            lev = level_of(d["priority"])
            for th in (0.05, 0.10, 0.15, 0.20, 0.30):
                for lvl, ln in ((1, "medium"), (2, "high")):
                    rows.append({"ML": mname, "LLM": lname, "rule": f"ML>={th:.2f} OR LLM>={ln}",
                                 **attack_metrics(y, (p >= th) | (lev >= lvl))})
            rows.append({"ML": mname, "LLM": lname, "rule": "cascade ML>=0.50; [0.05;0.50) -> LLM>=high",
                         **attack_metrics(y, (p >= 0.5) | ((p >= 0.05) & (lev >= 2)))})
    res = pd.DataFrame(rows).sort_values(["F1", "prec"], ascending=False)
    res.to_csv(S8 / "tandem_sweep_all_llms.csv", index=False, encoding="utf-8-sig")
    print("\nТОП-10 пар (разведка, пороги подобраны на той же выборке — оценка завышена):")
    print(res.head(10).to_string(index=False))


def build_tandem(name, p_ml, theta, d_llm, llm_level_min, ml_lat):
    lev = level_of(d_llm["priority"])
    by_ml, by_llm = p_ml >= theta, lev >= llm_level_min
    flag = by_ml | by_llm
    conf = np.maximum(p_ml, lev / 3.0)
    out = pd.DataFrame({"i": d_llm["i"], "label": d_llm["label"], "priority": levels_from_score(conf, flag),
                        "llm_priority": d_llm["priority"], "ml_proba": np.round(p_ml, 3),
                        "source": np.where(by_ml & by_llm, "ML+LLM", np.where(by_ml, "ML", np.where(by_llm, "LLM", ""))),
                        "latency_s": np.round(ml_lat + d_llm["latency_s"].values, 6)})
    for f in FIELDS:
        out[f] = d_llm[f].fillna("")
    out.to_csv(S8 / f"{name}.csv", index=False, encoding="utf-8-sig")
    m = attack_metrics(out["label"].ne("benign"), is_positive(out["priority"]))
    print(f"{name}: ML>={theta} OR LLM>={['low', 'medium', 'high', 'critical'][llm_level_min]} -> {m}")


def part_c(ml, thresholds, latency, llms, tandem1, tandem2):
    p = ml["LightGBM"]
    th = thresholds["LightGBM"]
    for name, key, lvl in (("tandem_1", tandem1, 2), ("tandem_2", tandem2, 1)):
        if key not in llms:
            print(f"{name}: нет файла {key}.csv в results/stage_4 — пропущен")
            continue
        build_tandem(name, p, th, llms[key], lvl, latency["LightGBM"])


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tandem1", default="llm__ollama__qwen3.5-4b-q4_K_M__few__freq")
    ap.add_argument("--tandem2", default="llm__polza__openai-gpt-6-sol__few__none")
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--skip-full-test", action="store_true")
    a = ap.parse_args()
    if not a.skip_ml:
        part_a(a.skip_full_test)
    ml = dict(np.load(S8 / "ml_probas.npz"))
    thresholds = json.load(open(S8 / "ml_thresholds.json"))
    latency = json.load(open(S8 / "ml_predict_latency.json"))
    y = pd.read_csv(SAMPLE)["label"].ne("benign").values
    llms = llm_results()
    if llms:
        part_b({k: v for k, v in ml.items() if "[" not in k}, llms, y)
        part_c(ml, thresholds, latency, llms, a.tandem1, a.tandem2)
    else:
        print("нет результатов LLM в results/stage_4/llm__*.csv — части B и C пропущены")


if __name__ == "__main__":
    main()
