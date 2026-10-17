# 51_merge_claude_llm_only.py
# Собирает ответы Claude (режим «просто LLM») в results/stage_4/llm__claude__claude-opus-5-5-manual__few__none.csv
# (формат как у 14_stage4_llm_runner.py). Время не измерялось: latency_s пусто.
import json, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import RESULTS, SAMPLE_CTX
S4 = RESULTS / "stage_4"
df = pd.read_csv(SAMPLE_CTX); ans = {}
for f in sorted((S4 / "claude_llm_answers").glob("part_*.txt")):
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\{.*\}", line)
        if m:
            try: d = json.loads(m.group(0)); ans[int(d["i"])] = d
            except Exception as e: print("пропуск:", line[:80], e)
rows = []
for i in range(len(df)):
    d = ans.get(i); r = df.iloc[i]
    ok = bool(d) and d.get("priority") in ("low", "medium", "high", "critical")
    rows.append({"i": i, "alert_id": r.get("alert_id", i), "label": r["label"], "priority": d["priority"] if ok else "error",
                 "what_happened": (d or {}).get("what_happened", ""), "why_suspicious": (d or {}).get("why_suspicious", ""),
                 "recommendation": (d or {}).get("recommendation", ""), "parse_ok": ok, "latency_s": "",
                 "raw": json.dumps(d, ensure_ascii=False) if d else "ERROR: нет ответа"})
res = pd.DataFrame(rows); out = S4 / "llm__claude__claude-opus-5-5-manual__few__none.csv"
res.to_csv(out, index=False, encoding="utf-8-sig")
print(f"ответов: {res.parse_ok.sum()} из {len(res)} -> {out.name}")
