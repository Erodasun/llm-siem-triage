# 14_stage4_llm_runner.py
# Единый прогон LLM на оценочной выборке (заменяет 14, 15, 19, 20, 22, 27, 27b).
# Один промпт для всех моделей (src/prompts.py), ответ JSON {priority, what_happened, why_suspicious, recommendation}.
# Примеры:
#   python scripts\14_stage4_llm_runner.py --backend ollama --model qwen3.5:4b-q4_K_M --shots few
#   python scripts\14_stage4_llm_runner.py --backend ollama --model qwen3.5:4b-q4_K_M --shots few --context freq
#   python scripts\14_stage4_llm_runner.py --backend polza --model openai/gpt-6-sol --model google/gemini-3-pro-preview
#   python scripts\14_stage4_llm_runner.py --backend tokify --model gpt-6-sol
# Ключи облака берутся из переменных окружения POLZA_API_KEY / TOKIFY_API_KEY.
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import chat_completion, is_positive, attack_metrics, ollama_generate, parse_llm_json
from src.config import SAMPLE, SAMPLE_CTX, stage_dir, utf8_console
from src.prompts import SYSTEM, build_prompt

COLUMNS = ["i", "alert_id", "label", "priority", "what_happened", "why_suspicious", "recommendation",
           "parse_ok", "latency_s", "raw"]


def result_path(backend, model, shots, context, sample2=False):
    slug = re.sub(r"[^A-Za-z0-9._\-]+", "-", model).strip("-")
    if sample2:  # вторая выборка (скрипт 44): результаты в stage_8, не смешиваются с основными
        return stage_dir(8) / f"llm2__{backend}__{slug}__{shots}__{context}.csv"
    return stage_dir(4) / f"llm__{backend}__{slug}__{shots}__{context}.csv"


def ask(backend, model, prompt):
    if backend == "ollama":
        return ollama_generate(model, prompt, system=SYSTEM, num_predict=500, num_ctx=4096)
    return chat_completion(backend, model, SYSTEM, prompt, max_tokens=600)


def run(backend, model, shots, context, limit, sample2=False):
    src = SAMPLE_CTX if context != "none" else SAMPLE
    if sample2:
        src = stage_dir(8) / ("experiment_sample_2_ctx.csv" if context != "none" else "experiment_sample_2.csv")
    df = pd.read_csv(src)
    if limit:
        df = df.head(limit)
    out = result_path(backend, model, shots, context, sample2)
    done = {}
    if out.exists():
        prev = pd.read_csv(out)
        done = {int(r["i"]): r.to_dict() for _, r in prev.iterrows() if r["priority"] != "error"}
        print(f"resume: {len(done)} готово, {len(df) - len(done)} осталось")
    rows = []
    for i, r in df.iterrows():
        if i in done:
            rows.append(done[i])
            continue
        t0 = time.perf_counter()
        try:
            raw = ask(backend, model, build_prompt(r, shots, context))
        except Exception as e:
            raw = f"ERROR: {e}"
        parsed = parse_llm_json(raw)
        rows.append({"i": i, "alert_id": r.get("alert_id", i), "label": r["label"],
                     "priority": parsed["priority"], "what_happened": parsed["what_happened"],
                     "why_suspicious": parsed["why_suspicious"], "recommendation": parsed["recommendation"],
                     "parse_ok": parsed["parse_ok"], "latency_s": round(time.perf_counter() - t0, 3), "raw": raw})
        print(f"[{i + 1:>3}/{len(df)}] {r['label']:<22} -> {parsed['priority']:<9} {rows[-1]['latency_s']:.1f}s")
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
    res = pd.DataFrame(rows, columns=COLUMNS)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    m = attack_metrics(res["label"].ne("benign"), is_positive(res["priority"]))
    errors = int((res["priority"] == "error").sum())
    print(f"{model} [{shots}/{context}] {m} errors={errors} latency={res['latency_s'].mean():.2f}s -> {out}")


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "polza", "tokify"], required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--shots", choices=["zero", "few"], default="few")
    ap.add_argument("--context", choices=["none", "freq", "anomaly"], default="none")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample2", action="store_true", help="вторая оценочная выборка (results/stage_8)")
    a = ap.parse_args()
    for model in a.model:
        run(a.backend, model, a.shots, a.context, a.limit, a.sample2)


if __name__ == "__main__":
    main()
