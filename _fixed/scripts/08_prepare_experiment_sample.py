# 08_prepare_experiment_sample.py
# Оценочная выборка из теста (по умолчанию 200 алертов) из четырёх групп.
# share = доля benign среди всех алертов с таким же description в полном датасете.
#   easy_attack   в окне атаки и share < TAU (тип редко встречается в фоне);
#   hard_attack   в окне атаки и TAU <= share < HARD_MAX (тип встречается и в атаках, и в фоне);
#   hard_benign   фон с тем же description, что у hard_attack (парные примеры: одинаковый текст, разная метка);
#   easy_benign   случайный фон.
# Алерты в окне атаки с share >= HARD_MAX (почти чистый фон, попавший во временное окно) не берутся.
# Метка label остаётся исходной (benign / тип атаки), поэтому label != benign по-прежнему означает «атака».
# Сохраняются time, alert_id (номер строки в test_alerts.csv) и group.
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FULL, SAMPLE, SEED, TEST, TEST_ATTACKS, utf8_console


def take_by_type(df, per_type, seed):
    return pd.concat([g.sample(min(per_type, len(g)), random_state=seed) for _, g in df.groupby("label")])


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--hard-max", type=float, default=0.95)
    ap.add_argument("--easy-per-type", type=int, default=12)
    ap.add_argument("--hard-per-type", type=int, default=8)
    ap.add_argument("--size", type=int, default=200)
    a = ap.parse_args()

    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(FULL, usecols=["description", "label"])
    share = full.groupby("description")["label"].agg(lambda s: (s == "benign").mean())
    del full

    df = pd.read_csv(TEST)
    df.insert(0, "alert_id", range(len(df)))
    df["benign_share"] = df["description"].map(share)
    window = df[df["label"].isin(TEST_ATTACKS)]
    easy = window[window["benign_share"] < a.tau]
    hard = window[(window["benign_share"] >= a.tau) & (window["benign_share"] < a.hard_max)]
    print(f"в окнах атак: {len(window)}; easy-кандидатов {len(easy)}, hard-кандидатов {len(hard)}")
    print("easy по типам:", easy["label"].value_counts().to_dict())
    print("hard по типам:", hard["label"].value_counts().to_dict())

    easy_attack = take_by_type(easy, a.easy_per_type, SEED).assign(group="easy_attack")
    hard_attack = take_by_type(hard, a.hard_per_type, SEED).assign(group="hard_attack")

    pool = df[df["label"] == "benign"]
    paired = []
    for desc, n in hard_attack["description"].value_counts().items():
        cand = pool[pool["description"] == desc]
        paired.append(cand.sample(min(n, len(cand)), random_state=SEED))
    hard_benign = pd.concat(paired).assign(group="hard_benign")

    n_easy_benign = a.size - len(easy_attack) - len(hard_attack) - len(hard_benign)
    rest = pool.drop(hard_benign.index)
    easy_benign = rest.sample(max(n_easy_benign, 0), random_state=SEED).assign(group="easy_benign")

    sample = pd.concat([easy_attack, hard_attack, hard_benign, easy_benign])
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.to_csv(SAMPLE, index=False)

    n_att = int(sample["label"].ne("benign").sum())
    p = n_att / len(sample)
    print(f"\nвыборка: {len(sample)} алертов, атак {n_att}, фона {len(sample) - n_att}")
    print(sample["group"].value_counts().to_string())
    print(pd.crosstab(sample["group"], sample["label"]).to_string())
    print("\nтипы событий в hard_attack:")
    print(hard_attack["description"].value_counts().head(8).to_string())
    print(f"\nбазовая линия «всё — атака»: F1 = {2 * p / (1 + p):.3f}")
    print("saved:", SAMPLE)


if __name__ == "__main__":
    main()
