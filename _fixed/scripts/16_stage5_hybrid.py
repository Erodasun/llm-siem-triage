# 16_stage5_hybrid.py
# Этап 5: схема «ML-фильтр». LightGBM на тексте (TF-IDF) даёт вероятность атаки; алерты с p < θ
# закрываются автоматически, остальные уходят дальше (LLM/аналитику).
# Для каждого θ на полном тесте считается доля отфильтрованного потока и доля потерянных атак.
# Выход: results/stage_5/hybrid_summary.csv (все пороги) — источник рисунка 9.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import alert_text
from src.config import SEED, TEST, TRAIN, stage_dir, utf8_console

TRAIN_LIMIT = 100000
THETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def main():
    utf8_console()
    tr = pd.read_csv(TRAIN, usecols=["description", "full_log", "label"])
    per_label = TRAIN_LIMIT // tr["label"].nunique()
    tr = pd.concat([g.sample(min(per_label, len(g)), random_state=SEED) for _, g in tr.groupby("label")])
    te = pd.read_csv(TEST, usecols=["description", "full_log", "label"])
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = tf.fit_transform(alert_text(tr)).astype(np.float32)
    Xte = tf.transform(alert_text(te)).astype(np.float32)
    clf = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(Xtr, tr["label"].ne("benign").astype(int).values)
    proba = clf.predict_proba(Xte)[:, 1]
    attack = te["label"].ne("benign").values
    rows = []
    for th in THETAS:
        filtered = proba < th
        rows.append({"theta": th, "filtered_pct": round(filtered.mean(), 4),
                     "attacks_lost": int((attack & filtered).sum()), "attacks_total": int(attack.sum()),
                     "lost_pct": round((attack & filtered).sum() / attack.sum(), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(stage_dir(5) / "hybrid_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
