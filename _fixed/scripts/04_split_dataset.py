# 04_split_dataset.py
# Хронологическое разбиение AIT-ADS на обучение/тест.
# Граница каждого сценария = время первого алерта фазы "webshell".
#   раньше границы -> train (фон + ранние фазы: scans/dirb/wpscan)
#   позже/равно    -> test  (фазы эксплуатации: webshell/cracking/rev_shell/priv_esc + фон после)
# Формат вывода: таблицы в консоль (tabulate) + графики в PNG.
import os, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FULL as IN, stage_dir

OUT_DIR = str(stage_dir(1))

USECOLS = ["scenario", "source", "time", "host", "ip",
           "rule_id", "level", "description", "full_log", "label"]
ATTACKS = ["network_scans", "service_scans", "wpscan", "dirb", "webshell",
           "cracking", "reverse_shell", "privilege_escalation",
           "service_stop", "dnsteal"]


def split_by_webshell(df):
    """Граница сценария = первый webshell-алерт. Возвращает списки train/test."""
    trains, tests = [], []
    bounds = []
    for sc, g in df.groupby("scenario"):
        ws = g[g["label"] == "webshell"]
        if len(ws) == 0:
            trains.append(g)
            tests.append(g.iloc[0:0])
            bounds.append((sc, None))
            continue
        b = ws["time"].min()
        bounds.append((sc, b))
        trains.append(g[g["time"] < b])
        tests.append(g[g["time"] >= b])
    return pd.concat(trains), pd.concat(tests), bounds


def main():
    print("load:", IN)
    df = pd.read_csv(IN, usecols=USECOLS)
    print("rows:", len(df))

    train, test, bounds = split_by_webshell(df)

    bounds_tbl = []
    for sc, b in bounds:
        if b:
            bounds_tbl.append([sc, pd.Timestamp(b, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")])
        else:
            bounds_tbl.append([sc, "-"])
    print("\n=== Webshell boundary per scenario ===")
    print(tabulate(bounds_tbl, headers=["scenario", "first webshell (UTC)"], tablefmt="psql"))

    scen_rows = []
    for sc in sorted(df["scenario"].unique()):
        tr = train[train["scenario"] == sc]
        te = test[test["scenario"] == sc]
        scen_rows.append([
            sc,
            len(tr), int((tr["label"] != "benign").sum()),
            len(te), int((te["label"] != "benign").sum()),
        ])
    print("\n=== Per-scenario split ===")
    print(tabulate(scen_rows, headers=["scenario", "train", "train_attack", "test", "test_attack"],
                   tablefmt="psql"))

    counts = pd.DataFrame({
        "train": train["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0).astype(int)
    counts = counts.reindex(["benign"] + ATTACKS).T
    print("\n=== Split by label (train / test) ===")
    print(tabulate(counts, headers="keys", tablefmt="psql"))

    summary = {"train": int(len(train)), "test": int(len(test)),
               "train_non_benign": int((train["label"] != "benign").sum()),
               "test_non_benign": int((test["label"] != "benign").sum())}
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    train.to_csv(os.path.join(OUT_DIR, "train_alerts.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test_alerts.csv"), index=False)
    counts_fmt = counts.reset_index().rename(columns={"index": "split"})
    counts_fmt.to_csv(os.path.join(OUT_DIR, "split_by_label.csv"), index=False)
    scen_df = pd.DataFrame(scen_rows, columns=["scenario", "train", "train_attack", "test", "test_attack"])
    scen_df.to_csv(os.path.join(OUT_DIR, "split_by_scenario.csv"), index=False)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("saved to", OUT_DIR)


    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(counts.columns))
    w = 0.38
    ax.bar(x - w/2, counts.loc["train"].values, width=w, label="train", color="#4C72B0")
    ax.bar(x + w/2, counts.loc["test"].values, width=w, label="test", color="#DD8452")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.columns, rotation=45, ha="right")
    ax.set_ylabel("alerts (log)")
    ax.set_title("Alert counts by label: train vs test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_label_distribution.png"), dpi=150)
    plt.close(fig)


    ss = scen_df.sort_values("scenario")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(range(len(ss)), ss["train"], label="train", color="#4C72B0")
    ax2.bar(range(len(ss)), ss["test"], bottom=ss["train"], label="test", color="#DD8452")
    ax2.set_xticks(range(len(ss)))
    ax2.set_xticklabels(ss["scenario"], rotation=45, ha="right")
    ax2.set_ylabel("alerts")
    ax2.set_title("Train/test sizes per scenario")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "02_scenario_split.png"), dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    main()