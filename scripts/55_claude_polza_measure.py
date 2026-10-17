# 55_claude_polza_measure.py — замер скорости и цены Claude Opus 5.5 через API Polza (проба на 5 алертах).
# Объяснения Claude для всех 102 алертов получены вручную (скрипты 48-49); здесь только время и цена.
# Время — из файла пробы (скрипт 47 --backend polza --limit 5), цена — по списанию с баланса Polza (до/после пробы).
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1] / "results" / "stage_7"
src = R / "tandem_main__polza__anthropic-claude-opus-5.5.csv"
d = pd.read_csv(src)
l = d[d["source"] == "ML+LLM"]
before, after = 81.18, 72.48   # баланс Polza, ₽ (по данным пользователя)
n = len(l)
out = {"model": "anthropic/claude-opus-5.5", "n_trial": n, "lat_s": round(float(l["latency_s"].mean()), 1),
       "lat_min": round(float(l["latency_s"].min()), 1), "lat_max": round(float(l["latency_s"].max()), 1),
       "spent_rub": round(before - after, 2), "rub_per_alert": round((before - after) / n, 2),
       "cost_102": round((before - after) / n * 102), "cost_200": round((before - after) / n * 200)}
json.dump(out, open(R / "claude_polza_measure.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(out)
