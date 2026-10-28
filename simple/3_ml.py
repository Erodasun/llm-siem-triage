# 3_ml.py — ШАГ 3. Сравнение пяти моделей машинного обучения и проверка, какие признаки помогают.
#
# Как обучаем (одинаково для всех моделей):
#   1) Два прогона: «проверка на A» и «проверка на B». В прогоне «проверка на A» модель учится на ранней части
#      всех 8 сетей и на поздней части сетей группы B, а оценивается на поздней части сетей группы A (их она не видела).
#   2) Из обучающих алертов каждой метки берём не больше 13 636 случайных (зерно 42), чтобы dirb и фон
#      не заглушили редкие атаки.
#   3) Ответ для обучения: 1 — любая атака, 0 — фон.
#   4) Признаки (src/features.py): текст, всплески одинаковых событий, активность хоста и источника, источник.
#      Словарь текста строится только по обучающим алертам.
#   5) Модель выдаёт вероятность атаки; порог 0,5 одинаковый для всех и не подбирается.
#   6) Итоговая модель выбирается заранее заданным правилом: лучшая average precision (AP) на всём тесте.
#   7) Для итоговой модели — проверка признаков: только текст; + всплески; + активность (основной набор);
#      + название сети; IP-адреса в тексте скрыты.
#   8) Итоговая модель обучается на всех данных и сохраняется для тандема и дашборда.
# Результат: simple/results/ml_comparison.csv, ablation.csv, by_scenario.csv, importance.csv, scores/*.npz, final_model.pkl
# Запуск: python simple/3_ml.py               (готовые прогоны из scores/ не пересчитываются; --fresh — всё заново)
import gc, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import GROUPS, LABELS_CSV, OUT, PER_LABEL, SEED, THETA, utf8
from src.features import MAIN_GROUPS, Featurizer, context_table
from src.metrics import attack_metrics, baseline, by_scenario, episodes, f1_interval, ranking

SCORES = OUT / "scores"
TEST_ATTACKS = ["webshell", "cracking", "reverse_shell", "privilege_escalation", "service_stop", "dnsteal"]


def models():
    return {
        "XGBoost": XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                                 tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=500, max_depth=7, learning_rate=0.08, subsample=0.9, subsample_freq=1,
                                   colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=SEED, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0),
        "LinearSVM": LinearSVC(max_iter=3000, random_state=SEED),
    }


ABLATION = {"только текст": ("text",),
            "текст + всплески": ("text", "bursts"),
            "текст + всплески + активность + источник (основной)": MAIN_GROUPS,
            "основной + название сети": MAIN_GROUPS + ("identity",),
            "основной, IP-адреса в тексте скрыты": MAIN_GROUPS + ("mask_ip",)}


def proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))       # у SVM нет вероятности: переводим расстояние в 0..1


class Data:
    """Таблица алертов без текста + текст по требованию + признаки контекста."""

    def __init__(self):
        cols = ["rid", "scenario", "source", "host", "time", "description", "label", "part", "group"]
        self.meta = pd.read_parquet(OUT / "alerts.parquet", columns=cols)
        for c in ("scenario", "source", "host", "description", "label", "part", "group"):
            self.meta[c] = self.meta[c].astype("category")          # экономия памяти
        self.logs = pq.read_table(OUT / "alerts.parquet", columns=["full_log"]).column(0)
        f = OUT / "context.parquet"
        if not f.exists():
            t0 = time.time()
            m = self.meta[["scenario", "source", "host", "time", "description"]].astype(
                {c: "category" for c in ("scenario", "source", "host", "description")})
            context_table(m).to_parquet(f)
            print(f"признаки контекста посчитаны за {time.time() - t0:.0f} с", flush=True)
        self.ctx = pd.read_parquet(f)
        self.test = self.meta[self.meta.part == "late"].sort_values(["scenario", "rid"], kind="stable").reset_index(drop=True)
        self.y = self.test["label"].ne("benign").values
        share = pd.read_csv(OUT / "share.csv", index_col=0)["benign_share"]
        self.dist = ~(self.y & (self.test["description"].map(share).values >= 0.95))   # без «неразличимых» атак
        self.samples = {k: pd.read_csv(OUT / f"{k}.csv", usecols=["alert_id", "label"]) for k in ("sample1", "sample2")}

    def rows(self, rids):
        df = self.meta.iloc[rids].copy()           # rid совпадает с номером строки
        df["full_log"] = self.logs.take(rids).to_pandas().values
        return df

    def pool(self, held):
        m = self.meta
        cand = m[(m.part == "early") | ((m.part == "late") & (m.group != held))]
        rids = np.concatenate([g["rid"].values if len(g) <= PER_LABEL else g.sample(PER_LABEL, random_state=SEED)["rid"].values
                               for _, g in cand.groupby("label")])
        return self.rows(np.sort(rids))


def run(data, name, make, groups):
    """Два прогона по группам → вероятность атаки для каждого тестового алерта от модели, которая его не видела."""
    f = SCORES / f"{name}.npy"
    if f.exists():
        return np.load(f)
    p = np.full(len(data.test), np.nan)
    for held in GROUPS:
        part_f = SCORES / f"{name}__fold{held}.npy"          # прогон одной группы сохраняется сразу
        idx = np.where(data.test["group"].values == held)[0]
        if part_f.exists():
            p[idx] = np.load(part_f)
            continue
        t0 = time.time()
        pool = data.pool(held)
        n_pool = len(pool)
        fz = Featurizer(groups).fit(pool)
        model = make().fit(fz.transform(pool, data.ctx), pool["label"].ne("benign").values.astype(int))
        del pool
        for i in range(0, len(idx), 20000):
            part = idx[i:i + 20000]
            p[part] = proba(model, fz.transform(data.rows(data.test["rid"].values[part]), data.ctx))
        np.save(part_f, p[idx])
        del model, fz
        gc.collect()
        print(f"  {name}: проверка на {held} готова за {time.time() - t0:.0f} с ({n_pool} обучающих алертов)", flush=True)
    np.save(f, p)
    return p


def evaluate(data, p):
    pred = p >= THETA
    r = {}
    for k, s in data.samples.items():
        ys, ps = s["label"].ne("benign").values, pred[s["alert_id"].values]
        lo, hi = f1_interval(ys, ps)
        r[k] = {**attack_metrics(ys, ps), "F1_low": lo, "F1_high": hi}
    r["stream"] = {**attack_metrics(data.y, pred), **ranking(data.y, p)}
    r["distinguishable"] = attack_metrics(data.y[data.dist], pred[data.dist])
    lab = pd.read_csv(LABELS_CSV)
    r["episodes"] = episodes(data.test, pred, lab[lab["attack"].isin(TEST_ATTACKS)])
    return r


def row(name, r):
    s1, s2, st, d, e = r["sample1"], r["sample2"], r["stream"], r["distinguishable"], r["episodes"]
    return {"модель": name,
            "выборка 1: найдено": f'{s1["found"]} из {s1["attacks"]}', "выборка 1: ложных": s1["fp"],
            "выборка 1: F1": s1["F1"], "выборка 1: F1 95% ДИ": f'{s1["F1_low"]}–{s1["F1_high"]}',
            "выборка 2: найдено": f'{s2["found"]} из {s2["attacks"]}', "выборка 2: ложных": s2["fp"], "выборка 2: F1": s2["F1"],
            "поток: найдено": st["found"], "поток: ложных": st["fp"], "поток: F1": st["F1"],
            "поток: ROC-AUC": st["ROC_AUC"], "поток: AP": st["AP"],
            "различимые: найдено": f'{d["found"]} из {d["attacks"]}',
            "эпизоды": f'{e["found"]} из {e["episodes"]}', "тревог в час": e["flagged_per_hour"]}


def main():
    utf8()
    SCORES.mkdir(parents=True, exist_ok=True)
    if "--fresh" in sys.argv:
        for f in list(SCORES.glob("*.npy")) + [OUT / "final_model.pkl"]:
            f.unlink(missing_ok=True)
    data = Data()
    print(f"тест: {len(data.test)} алертов, атак {int(data.y.sum())}", flush=True)

    # 1) пять моделей на основном наборе признаков
    res, rows = {}, []
    for name, make in ((n, (lambda n=n: models()[n])) for n in models()):
        res[name] = evaluate(data, run(data, name, make, MAIN_GROUPS))
        rows.append(row(name, res[name]))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    base = {k: baseline(s["label"].ne("benign").values) for k, s in data.samples.items()}
    base["stream"] = baseline(data.y)
    rows.append({"модель": "«всё — атака» (базовая линия)", "выборка 1: F1": base["sample1"]["F1"],
                 "выборка 2: F1": base["sample2"]["F1"], "поток: F1": base["stream"]["F1"]})
    pd.DataFrame(rows).to_csv(OUT / "ml_comparison.csv", index=False, encoding="utf-8-sig")
    best = max(res, key=lambda n: res[n]["stream"]["AP"])
    print(f"\nитоговая модель (лучшая AP на всём тесте): {best}", flush=True)

    # 2) какие признаки помогают итоговой модели
    rows = []
    for title, groups in ABLATION.items():
        extra = [g for g in groups if g not in MAIN_GROUPS]
        tag = best if groups == MAIN_GROUPS else (f"{best}__{extra[0]}" if extra == ["mask_ip"] else f"{best}__{'+'.join(groups)}")
        rows.append(row(title, evaluate(data, run(data, tag, lambda: models()[best], groups))))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False, encoding="utf-8-sig")

    # 3) качество итоговой модели по сетям
    p = np.load(SCORES / f"{best}.npy")
    by_scenario(data.test, data.y, p).to_csv(OUT / "by_scenario.csv", index=False)

    # 4) итоговая модель на всех данных: для тандема, дашборда и замера скорости
    if (OUT / "final_model.pkl").exists() and (OUT / "ml_results.json").exists():
        r = json.load(open(OUT / "ml_results.json", encoding="utf-8"))
        r.update({"results": res, "baseline": base})
        json.dump(r, open(OUT / "ml_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        print("итоговая модель уже обучена (final_model.pkl); для переобучения запустите с --fresh")
        return
    m = data.meta
    rids = np.concatenate([g["rid"].values if len(g) <= PER_LABEL else g.sample(PER_LABEL, random_state=SEED)["rid"].values
                           for _, g in m.groupby("label")])
    pool = data.rows(np.sort(rids))
    fz = Featurizer(MAIN_GROUPS).fit(pool)
    model = models()[best].fit(fz.transform(pool, data.ctx), pool["label"].ne("benign").values.astype(int))
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        pd.Series(imp, index=fz.names()).sort_values(ascending=False).head(30).to_csv(OUT / "importance.csv", header=["важность"])
    s1 = data.rows(data.test["rid"].values[data.samples["sample1"]["alert_id"].values])
    t0 = time.perf_counter()
    for _ in range(5):
        proba(model, fz.transform(s1, data.ctx))
    ms = (time.perf_counter() - t0) / 5 / len(s1) * 1000
    pickle.dump({"name": best, "model": model, "featurizer": fz}, open(OUT / "final_model.pkl", "wb"))
    json.dump({"best": best, "ms_per_alert_incl_features": round(ms, 3), "results": res, "baseline": base},
              open(OUT / "ml_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"готово; время на алерт с подготовкой признаков: {ms:.3f} мс", flush=True)


if __name__ == "__main__":
    main()
