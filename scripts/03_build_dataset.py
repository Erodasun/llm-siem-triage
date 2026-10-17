# 03_build_dataset.py
# Сбор рабочего датасета AIT-ADS с временной разметкой (включает бывший 01_assign_labels.py).
# Выход: data/processed/02_labeled_alerts.csv, data/processed/01_label_counts.csv
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED, DATA_RAW, FULL, LABELS_CSV, SCENARIOS

HEADERS = ["scenario", "source", "time", "host", "ip", "rule_id", "level", "description", "full_log", "label"]


def load_attacks():
    attacks = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            attacks.setdefault(row["scenario"], []).append(
                (row["attack"], float(row["start"]), float(row["end"])))
    return attacks


def alert_time(obj):
    ld = obj.get("LogData", {})
    if ld.get("DetectionTimestamp"):
        return ld["DetectionTimestamp"][0]
    if ld.get("Timestamps"):
        return ld["Timestamps"][0]
    ts = obj.get("@timestamp")
    if ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def is_training(obj):
    ac = obj.get("AnalysisComponent", {})
    return isinstance(ac, dict) and ac.get("TrainingMode") is True


def label_alert(t, attack_list):
    for name, start, end in attack_list:
        if start <= t <= end:
            return name
    return "benign"


def flat(obj, scenario, t, attacks):
    ld = obj.get("LogData", {})
    raw = ld.get("RawLogData")
    if raw is None:
        raw = obj.get("full_log", "")
    elif isinstance(raw, list):
        raw = " ".join(raw)
    if "LogData" in obj:
        ac = obj.get("AnalysisComponent", {})
        source, description = "aminer", ac.get("AnalysisComponentName", "") or ""
        host, ip, rule_id, level = "", obj.get("AMiner", {}).get("ID", "") or "", "", ""
    else:
        rule, agent = obj.get("rule", {}), obj.get("agent", {})
        source, description = "wazuh", rule.get("description", "") or ""
        host, ip = agent.get("name", "") or "", agent.get("ip", "") or ""
        rule_id, level = str(rule.get("id", "")), rule.get("level", "") or ""
    return [scenario, source, t, host, ip, rule_id, level, description, raw, label_alert(t, attacks[scenario])]


def main():
    attacks = load_attacks()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    counts = {sc: Counter() for sc in SCENARIOS}
    no_time = bad_lines = total = 0
    with open(FULL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for sc in SCENARIOS:
            for suffix in ("aminer", "wazuh"):
                with open(DATA_RAW / f"{sc}_{suffix}.json", encoding="utf-8") as fr:
                    for line in fr:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            bad_lines += 1
                            continue
                        if is_training(obj):
                            continue
                        t = alert_time(obj)
                        if t is None:
                            no_time += 1
                            continue
                        row = flat(obj, sc, t, attacks)
                        w.writerow(row)
                        counts[sc][row[-1]] += 1
                        total += 1
            print(f"{sc:<16}{sum(counts[sc].values()):>10}  attack={sum(counts[sc].values()) - counts[sc]['benign']}")
    with open(DATA_PROCESSED / "01_label_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "label", "count"])
        for sc in SCENARIOS:
            for lab, n in counts[sc].items():
                w.writerow([sc, lab, n])
    print(f"total={total} no_time={no_time} bad_lines={bad_lines}")
    print("saved:", FULL)


if __name__ == "__main__":
    main()
