# 49_merge_claude_answers.py
# Собирает ответы Claude из results/stage_7/claude_answers/part_*.txt в tandem_main__claude__manual.csv
# (формат как у 47_tandem_explain.py). Время не измерялось: latency_s = время ML.
import json, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import RESULTS
S7 = RESULTS / "stage_7"
ml = pd.read_csv(S7 / "ml_only_lightgbm.csv").sort_values("i").reset_index(drop=True)
lat = json.load(open(S7 / "ml_predict_latency.json"))["LightGBM"]
ans = {}
for f in sorted((S7 / "claude_answers").glob("part_*.txt")):
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\{.*\}", line)
        if m:
            try: d = json.loads(m.group(0)); ans[int(d["i"])] = d
            except Exception as e: print("пропуск строки:", line[:80], e)
rows = []
for i in range(len(ml)):
    b = {"i": i, "label": ml.at[i, "label"], "ml_proba": ml.at[i, "ml_proba"]}
    if ml.at[i, "priority"] == "low":
        rows.append({**b, "llm_model": "", "priority": "low", "source": "ML", "attack_signs": "", "latency_s": round(lat, 6),
                     "what_happened": "", "why_suspicious": "", "recommendation": "", "raw": ""}); continue
    d = ans.get(i, {})
    rows.append({**b, "llm_model": "claude (вручную)", "priority": d.get("priority", "medium"), "source": "ML+LLM",
                 "attack_signs": d.get("attack_signs", ""), "latency_s": round(lat, 6),
                 "what_happened": d.get("what_happened", ""), "why_suspicious": d.get("why_suspicious", ""),
                 "recommendation": d.get("recommendation", ""), "raw": json.dumps(d, ensure_ascii=False) if d else "ERROR: нет ответа"})
out = pd.DataFrame(rows); out.to_csv(S7 / "tandem_main__claude__manual.csv", index=False, encoding="utf-8-sig")
fl = out[out.source == "ML+LLM"]
print(f"объяснений: {(fl.what_happened != '').sum()} из {len(fl)}; без признаков атаки: {(fl.attack_signs == 'нет').sum()}")
