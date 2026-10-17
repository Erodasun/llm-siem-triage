# 47_tandem_explain.py
# Главный тандем: LightGBM отбирает алерты → LLM объясняет только отобранные.
# Два варианта пары: с локальной Qwen 3.5 4B (Ollama, бесплатно) и с облачной GPT-6-sol (Polza AI, платно).
#   1) Решение «атака / не атака» принимает LightGBM (порог θ с валидации, файл results/stage_7/ml_only_lightgbm.csv).
#   2) Алерты, которые LightGBM признал атакой (приоритет medium и выше), по одному отправляются в локальную LLM.
#      LLM получает задание: «классификатор признал алерт атакой — объясни, что произошло, почему это опасно,
#      что делать аналитику, и оцени срочность (medium / high / critical)». Отменить решение ML она не может,
#      но может честно отметить, что признаков атаки в тексте не видит (поле attack_signs = нет).
#   3) Остальные алерты получают приоритет low и в LLM не отправляются — на них не тратится время.
# Число найденных атак и ложных тревог у тандема то же, что у LightGBM (решает ML); LLM добавляет объяснение.
# Запуск:
#   python scripts\47_tandem_explain.py                                   (Qwen 3.5 4B локально, бесплатно)
#   python scripts\47_tandem_explain.py --backend polza --model openai/gpt-6-sol --limit 5   (облако, проба на 5 алертах)
#   python scripts\47_tandem_explain.py --backend polza --model openai/gpt-6-sol             (облако, все отобранные)
# Ключ облака — только из переменной окружения POLZA_API_KEY.
# Можно прерывать: при повторном запуске готовые алерты пропускаются (повторно не оплачиваются).
# Результат: results/stage_7/tandem_main__<backend>__<модель>.csv
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import chat_completion, log_snippet, ollama_generate, parse_llm_json
from src.config import RESULTS, SAMPLE_CTX, utf8_console
from src.prompts import context_line

S7 = RESULTS / "stage_7"


def out_path(backend, model):
    slug = re.sub(r"[^A-Za-z0-9._\-]+", "-", model).strip("-")
    return S7 / f"tandem_main__{backend}__{slug}.csv"


def ask(backend, model, prompt):
    if backend == "ollama":
        return ollama_generate(model, prompt, system=SYSTEM, num_predict=500, num_ctx=4096)
    return chat_completion(backend, model, SYSTEM, prompt, max_tokens=600)

SYSTEM = (
    "Ты — старший аналитик SOC. Классификатор машинного обучения уже отобрал этот алерт SIEM как вероятную атаку "
    "и передал его тебе. Твоя задача — объяснить аналитику-человеку, что произошло, почему это может быть опасно "
    "и что сделать в первую очередь, и оценить срочность. Пиши по-русски, коротко и конкретно, опираясь на текст "
    "алерта и фрагмент журнала. Если в тексте нет явных признаков атаки, честно напиши об этом в why_suspicious и "
    "поставь attack_signs = \"нет\". Ответь одним JSON-объектом строго с ключами "
    '"priority", "attack_signs", "what_happened", "why_suspicious", "recommendation". '
    "priority — только одно из слов medium, high, critical; attack_signs — только \"да\" или \"нет\". "
    "Больше ничего не выводи."
)

EXAMPLE = (
    "Пример: описание='Reverse Shell'; детали='session opened with bash -i'; вероятность атаки по ML 0,99.\n"
    'Ответ: {"priority": "critical", "attack_signs": "да", '
    '"what_happened": "На хосте открыта обратная командная оболочка — злоумышленник получил интерактивный доступ", '
    '"why_suspicious": "bash -i с подключением наружу — прямой признак компрометации", '
    '"recommendation": "Разорвать соединение, изолировать хост, найти способ запуска оболочки по журналам"}\n\n'
)


def build_prompt(row, proba):
    return (EXAMPLE + "Теперь объясни этот алерт.\n"
            f"Описание: {row['description']}\n"
            f"Детали: {log_snippet(row['full_log'], 400)}\n"
            f"Контекст: {context_line(row, 'freq')}\n"
            f"Вероятность атаки по ML: {proba:.2f}\n")


def parse(raw):
    p = parse_llm_json(raw)
    if p["priority"] not in ("medium", "high", "critical"):
        p["priority"] = "medium" if p["priority"] in ("low", "error") else p["priority"]
    m = re.search(r'"attack_signs"\s*:\s*"?\s*(да|нет|yes|no)', raw or "", re.I)
    p["attack_signs"] = ("да" if m.group(1).lower() in ("да", "yes") else "нет") if m else ""
    return p


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "polza"], default="ollama")
    ap.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    ap.add_argument("--limit", type=int, default=0, help="для пробного запуска: объяснить только первые N отобранных")
    a = ap.parse_args()
    OUT = out_path(a.backend, a.model)
    legacy = S7 / "tandem_main.csv"  # первая версия скрипта писала сюда результаты Qwen
    if a.backend == "ollama" and legacy.exists() and not OUT.exists():
        legacy.rename(OUT)

    sample = pd.read_csv(SAMPLE_CTX)
    ml = pd.read_csv(S7 / "ml_only_lightgbm.csv").sort_values("i").reset_index(drop=True)
    ml_lat = json.load(open(S7 / "ml_predict_latency.json"))["LightGBM"]
    flag = ml["priority"].ne("low").values
    print(f"LightGBM отобрал {flag.sum()} из {len(ml)} алертов; они уйдут в {a.model}")

    done = {}
    if OUT.exists():
        prev = pd.read_csv(OUT)
        ok = prev["source"].eq("ML") | ~prev["raw"].fillna("").astype(str).str.startswith("ERROR")
        done = {int(r["i"]): r.to_dict() for _, r in prev[ok].iterrows()}
        print(f"продолжение: {len(done)} строк уже готово")

    rows, n_llm = [], 0
    for i in range(len(ml)):
        if i in done:
            rows.append(done[i]); continue
        base = {"i": i, "label": ml.at[i, "label"], "ml_proba": ml.at[i, "ml_proba"], "llm_model": ""}
        if not flag[i]:
            rows.append({**base, "priority": "low", "source": "ML", "attack_signs": "", "latency_s": round(ml_lat, 6),
                         "what_happened": "", "why_suspicious": "", "recommendation": "", "raw": ""})
            continue
        if a.limit and n_llm >= a.limit:
            continue
        t0 = time.perf_counter()
        try:
            raw = ask(a.backend, a.model, build_prompt(sample.iloc[i], float(ml.at[i, "ml_proba"])))
        except Exception as e:
            raw = f"ERROR: {e}"
        p = parse(raw)
        dt = time.perf_counter() - t0
        n_llm += 1
        rows.append({**base, "priority": p["priority"], "source": "ML+LLM", "llm_model": a.model,
                     "attack_signs": p["attack_signs"], "latency_s": round(ml_lat + dt, 3),
                     "what_happened": p["what_happened"], "why_suspicious": p["why_suspicious"],
                     "recommendation": p["recommendation"], "raw": raw})
        print(f"[{n_llm:>3}/{flag.sum()}] {base['label']:<22} -> {p['priority']:<8} признаки атаки: {p['attack_signs'] or '?':<3} {dt:.1f}s")
        if n_llm % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    llm_rows = d[d["source"] == "ML+LLM"]
    bad = llm_rows["raw"].astype(str).str.startswith("ERROR").sum() + (llm_rows["what_happened"].fillna("") == "").sum()
    print(f"готово: {len(llm_rows)} объяснений, без объяснения/с ошибкой: {bad}; "
          f"LLM не видит признаков атаки: {(llm_rows['attack_signs'] == 'нет').sum()}; "
          f"среднее время LLM на отобранный алерт: {(llm_rows['latency_s'] - ml_lat).mean():.1f} с -> {OUT}")


if __name__ == "__main__":
    main()
