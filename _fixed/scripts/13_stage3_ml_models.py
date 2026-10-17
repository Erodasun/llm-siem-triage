# 13_stage3_ml_models.py
# Этап 3: сравнение классических ML моделей.
# RandomForest, XGBoost, LightGBM, LogisticRegression, LinearSVM.
# Обучение: страт. выборка из train (100k). Оценка: полный test (156 592).
# Метрики: precision/recall/F1 по атакам (предсказание "не-benign"),
#          accuracy, время обучения, время предсказания.
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SEED, TEST, TRAIN, stage_dir
from src.common import alert_text

OUT_DIR = str(stage_dir(3))
TRAIN_LIMIT = 100000
COLUMNS = ["model", "train_s", "pred_s", "precision", "recall", "F1", "accuracy"]


def main():
    print("loading train...")
    tr = pd.read_csv(TRAIN, usecols=["description", "full_log", "label"])
    parts = []
    for lab, g in tr.groupby("label"):
        parts.append(g.sample(min(TRAIN_LIMIT // tr["label"].nunique(), len(g)), random_state=SEED))
    tr = pd.concat(parts, ignore_index=True)
    tr["text"] = alert_text(tr)
    print("train sample:", len(tr))

    print("loading test...")
    te = pd.read_csv(TEST, usecols=["description", "full_log", "label"])
    te["text"] = alert_text(te)

    tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = tf.fit_transform(tr["text"]).astype(np.float32)
    Xte = tf.transform(te["text"]).astype(np.float32)
    ytr = tr["label"].ne("benign").astype(int).values
    yte = te["label"].ne("benign").astype(int).values

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, n_jobs=-1),
        "LinearSVM": LinearSVC(max_iter=3000, random_state=SEED),
    }

    rows = []
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        t_train = time.time() - t0
        t0 = time.time()
        pred = m.predict(Xte)
        t_pred = time.time() - t0
        prec = precision_score(yte, pred, zero_division=0)
        rec = recall_score(yte, pred, zero_division=0)
        f1 = f1_score(yte, pred, zero_division=0)
        acc = accuracy_score(yte, pred)
        rows.append([name, round(t_train, 2), round(t_pred, 3),
                     round(prec, 4), round(rec, 4), round(f1, 4), round(acc, 4)])
        print(f"done: {name}")
    print("\n=== Etap 3: 5 ML models on full test (156 592 alerts) ===")
    print(tabulate(rows, headers=COLUMNS, tablefmt="psql"))
    base = yte.mean()
    print(f"trivial 'all attack': precision={base:.4f} recall=1.0 F1={2*base/(1+base):.4f}")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(os.path.join(OUT_DIR, "ml_models_comparison.csv"), index=False)
    print("saved:", os.path.join(OUT_DIR, "ml_models_comparison.csv"))


if __name__ == "__main__":
    main()