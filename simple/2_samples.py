# 2_samples.py — ШАГ 2. Две оценочные выборки по 200 алертов из поздней части (теста).
#
# Зачем: языковые модели медленные и платные, на всех 156 592 алертах их не проверить.
# На этих же 200 алертах сравниваются и ML-модели, и LLM.
#
# «Различимость»: для каждого названия события считаем долю фона среди всех алертов с таким названием (share).
#   easy_attack — атака, текст которой почти не бывает в фоне (share < 0,5);
#   hard_attack — атака, текст которой бывает и в фоне (0,5 ≤ share < 0,95);
#   hard_benign — фон с тем же текстом, что у hard_attack (самый трудный случай);
#   easy_benign — случайный фон.
# Атаки с share ≥ 0,95 (почти чистый фон, попавший во временное окно атаки) в выборки не берутся.
# Выборка 1 — зерно 42; выборка 2 — зерно 7 и без алертов выборки 1.
# Результат: simple/results/sample1.csv, sample2.csv, share.csv
# Запуск: python simple/2_samples.py
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import OUT, utf8

TEST_ATTACKS = ["webshell", "cracking", "reverse_shell", "privilege_escalation", "service_stop", "dnsteal"]


def make_sample(test, seed, exclude=(), size=200, tau=0.5, hard_max=0.95, easy_per_type=12, hard_per_type=8):
    df = test[~test["alert_id"].isin(set(exclude))]
    window = df[df["label"].isin(TEST_ATTACKS)]
    easy = window[window["benign_share"] < tau]
    hard = window[(window["benign_share"] >= tau) & (window["benign_share"] < hard_max)]
    take = lambda d, n: pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in d.groupby("label")])
    ea = take(easy, easy_per_type).assign(group="easy_attack")
    ha = take(hard, hard_per_type).assign(group="hard_attack")
    pool = df[df["label"] == "benign"]
    hb = pd.concat([pool[pool["description"] == d].sample(min(n, (pool["description"] == d).sum()), random_state=seed)
                    for d, n in ha["description"].value_counts().items()]).assign(group="hard_benign")
    eb = pool.drop(hb.index).sample(max(size - len(ea) - len(ha) - len(hb), 0), random_state=seed).assign(group="easy_benign")
    return pd.concat([ea, ha, hb, eb]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    utf8()
    a = pd.read_parquet(OUT / "alerts.parquet")
    share = (a["label"] == "benign").groupby(a["description"]).mean().rename("benign_share")
    share.to_csv(OUT / "share.csv")
    # тест в том же порядке, что и раньше: по сетям в алфавитном порядке, внутри — по исходному порядку строк
    test = a[a["part"] == "late"].sort_values(["scenario", "rid"], kind="stable").reset_index(drop=True)
    del a
    test.insert(0, "alert_id", range(len(test)))
    test["benign_share"] = test["description"].map(share)
    s1 = make_sample(test, 42)
    s2 = make_sample(test, 7, exclude=s1["alert_id"])
    for name, s in (("sample1", s1), ("sample2", s2)):
        s.to_csv(OUT / f"{name}.csv", index=False)
        n = int(s["label"].ne("benign").sum())
        p = n / len(s)
        print(f"{name}: {len(s)} алертов, атак {n}; «всё — атака» F1 = {2 * p / (1 + p):.3f}")
        print(s["group"].value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
