# 50_export_claude_llm_only.py
# Пакеты для Claude в режиме «просто LLM» (все 200 алертов, тот же промпт, что у GPT-6-sol: few-shot, без контекста,
# без вероятности ML и без меток). Результат: results/stage_4/claude_llm_batches/part_N.txt
# Ответы Claude (по JSON-строке на алерт) сохранить в results/stage_4/claude_llm_answers/part_N.txt, затем запустить 51.
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import RESULTS, SAMPLE_CTX
from src.prompts import SYSTEM, FEW_SHOT, build_prompt
OUT = RESULTS / "stage_4" / "claude_llm_batches"; OUT.mkdir(exist_ok=True)
df = pd.read_csv(SAMPLE_CTX)
HEAD = (SYSTEM + "\n\nНиже пакет алертов. Для КАЖДОГО верни ровно одну строку JSON (без пояснений вокруг) с ключами "
        '"i", "priority", "what_happened", "why_suspicious", "recommendation". Поле i — номер алерта из заголовка. '
        "Одна строка = один алерт, без переносов внутри строки.\n\n" + FEW_SHOT.replace("Ответ:", "Ответ (i опущено в примере):") + "\n")
N = 8; size = -(-len(df) // N)
for k in range(N):
    part = range(k * size, min(len(df), (k + 1) * size))
    body = "\n".join(f"=== АЛЕРТ i={i} ===\n" + build_prompt(df.iloc[i], "zero", "none") for i in part)
    (OUT / f"part_{k+1}.txt").write_text(HEAD + body, encoding="utf-8")
    print(k + 1, len(part))
