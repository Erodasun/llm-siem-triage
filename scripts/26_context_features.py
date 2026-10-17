# 26_context_features.py
# Частотный контекст для оценочной выборки (заменяет 26 и 26b).
#   burst_W - сколько таких же событий (scenario + description) было за последние W секунд, включая текущее
#   bg_W    - медианное число таких же событий в окне W по ПРОШЛОЙ истории этой группы (без заглядывания в будущее)
#   ratio_W - burst_W / bg_W
# Время берётся из самой выборки (колонка time), а не джойном по тексту.
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import burst_counts, time_index
from src.config import FULL, SAMPLE, SAMPLE_CTX, stage_dir, utf8_console

WINDOWS = [60, 300, 900]
KEYS = ("scenario", "description")


def past_background(arr, t, window):
    past = arr[arr < t]
    if len(past) == 0:
        return 1.0
    counts = np.searchsorted(past, past + window, side="right") - np.arange(len(past))
    return max(float(np.median(counts)), 1.0)


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample2", action="store_true", help="вторая оценочная выборка (results/stage_8)")
    a = ap.parse_args()
    src, dst = (stage_dir(8) / "experiment_sample_2.csv", stage_dir(8) / "experiment_sample_2_ctx.csv") if a.sample2 \
        else (SAMPLE, SAMPLE_CTX)
    s = pd.read_csv(src)
    if "time" not in s.columns:
        raise SystemExit("в выборке нет колонки time — перезапустите 08_prepare_experiment_sample.py")
    full = pd.read_csv(FULL, usecols=["scenario", "time", "description"])
    index = time_index(full, KEYS)
    keys = list(zip(s["scenario"], s["description"]))
    attack = s["label"].ne("benign")
    for w in WINDOWS:
        s[f"burst_{w}"] = burst_counts(index, keys, s["time"].values, w)
        s[f"bg_{w}"] = [past_background(index.get(k, np.array([])), t, w) for k, t in zip(keys, s["time"])]
        s[f"ratio_{w}"] = (s[f"burst_{w}"] / s[f"bg_{w}"]).round(2)
        print(f"W={w:>4}s burst median: attack={s.loc[attack, f'burst_{w}'].median():.0f} "
              f"benign={s.loc[~attack, f'burst_{w}'].median():.0f} | ratio median: "
              f"attack={s.loc[attack, f'ratio_{w}'].median():.2f} benign={s.loc[~attack, f'ratio_{w}'].median():.2f}")
    s.to_csv(dst, index=False)
    print("saved:", dst)


if __name__ == "__main__":
    main()
