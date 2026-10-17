# 48_export_claude_prompts.py
# Готовит пакеты алертов для объяснения в отдельном окне Claude (третья пара «LightGBM → Claude»).
# Метки (атака/не атака) в пакеты не попадают. Результат: results/stage_7/claude_batches/part_N.txt
# Ответы Claude (JSON-строки) сохранить в results/stage_7/claude_answers/part_N.txt, затем собрать скриптом 49.
import importlib.util, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import RESULTS, SAMPLE_CTX
spec = importlib.util.spec_from_file_location("t47", ROOT / "scripts" / "47_tandem_explain.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
S7 = RESULTS / "stage_7"; OUT = S7 / "claude_batches"; OUT.mkdir(exist_ok=True)
sample = pd.read_csv(SAMPLE_CTX)
ml = pd.read_csv(S7 / "ml_only_lightgbm.csv").sort_values("i").reset_index(drop=True)
idx = [i for i in range(len(ml)) if ml.at[i, "priority"] != "low"]
HEAD = (t.SYSTEM + "\n\nНиже пакет алертов. Для КАЖДОГО верни ровно одну строку JSON (без пояснений вокруг) с ключами "
        '"i", "priority", "attack_signs", "what_happened", "why_suspicious", "recommendation". '
        "Поле i — номер алерта из заголовка. Одна строка = один алерт, без переносов внутри строки.\n\n" + t.EXAMPLE)
N = 5; size = -(-len(idx) // N)
for k in range(N):
    part = idx[k * size:(k + 1) * size]
    body = "\n".join(f"=== АЛЕРТ i={i} ===\n" + t.build_prompt(sample.iloc[i], float(ml.at[i, 'ml_proba'])).split("Теперь объясни этот алерт.\n")[1] for i in part)
    (OUT / f"part_{k+1}.txt").write_text(HEAD + body, encoding="utf-8")
    print(k + 1, len(part))
