# metrics.py — как считаем качество.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def attack_metrics(y, pred):
    """Найдено атак, ложные тревоги, полнота, точность, F1."""
    y, p = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    rec, prec = tp / max(1, tp + fn), tp / max(1, tp + fp)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"found": tp, "attacks": int(y.sum()), "fp": fp, "recall": round(rec, 3),
            "precision": round(prec, 3), "F1": round(f1, 3)}


def baseline(y):
    """«Назвать атакой всё»."""
    return attack_metrics(y, np.ones(len(y), bool))


def f1_interval(y, pred, n=2000, seed=42):
    """95 % доверительный интервал F1: 2000 раз пересобираем выборку с возвращением."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        vals.append(attack_metrics(y[s], pred[s])["F1"])
    return round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)


def ranking(y, p):
    """Оценки, не зависящие от порога."""
    return {"ROC_AUC": round(roc_auc_score(y, p), 3), "AP": round(average_precision_score(y, p), 3)}


def episodes(test, flag, labels):
    """Эпизод атаки (интервал из labels.csv) найден, если отобран хотя бы один его алерт."""
    found, n, delays = 0, 0, []
    for r in labels.itertuples():
        m = (test.scenario.values == r.scenario) & (test.time.values >= r.start) & (test.time.values <= r.end)
        if not m.any():
            continue
        n += 1
        hit = m & flag
        if hit.any():
            found += 1
            delays.append(float(test.time.values[hit].min() - r.start))
    hours = sum((g.time.max() - g.time.min()) / 3600 for _, g in test.groupby("scenario"))
    return {"episodes": n, "found": found, "median_delay_s": round(float(np.median(delays)), 1) if delays else None,
            "alerts_per_hour": round(len(test) / hours, 1), "flagged_per_hour": round(float(flag.sum()) / hours, 1)}


def by_scenario(test, y, p):
    rows = []
    for sc, idx in test.groupby("scenario").indices.items():
        if y[idx].sum() and (~y[idx]).sum():
            rows.append({"scenario": sc, "attacks": int(y[idx].sum()), "ROC_AUC": round(roc_auc_score(y[idx], p[idx]), 3)})
    return pd.DataFrame(rows)
