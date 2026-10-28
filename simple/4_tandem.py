# 4_tandem.py — ШАГ 4. Тандем: итоговая ML-модель отбирает алерты, языковая модель объясняет только отобранные.
#
# Как работает на выборке 1 (200 алертов):
#   1) Итоговая модель из шага 3 даёт вероятность атаки; алерт отобран, если вероятность ≥ 0,5.
#   2) Не отобранные алерты получают приоритет low, языковая модель их не видит.
#   3) Для отобранных объяснение запрашивается у языковой модели; ответы сохраняются сразу и повторно не оплачиваются.
#   4) Четыре языковые модели: Qwen 3.5 4B (локально), GPT-6-sol и gpt-oss-120b (облако Polza AI),
#      Claude (вручную через окно чата).
# Результат: simple/results/tandem/<модель>.csv — 200 строк: решение ML, приоритет и объяснение.
# Сейчас REUSE_OLD = False: все объяснения получены заново с вероятностью итоговой модели в промпте.
#
# Запуск:
#   python simple/4_tandem.py                                    — собрать файлы и показать, каких объяснений не хватает
#   python simple/4_tandem.py --ask "gpt-oss-120b"                — дозапросить недостающие у модели (платно для облака)
#   python simple/4_tandem.py --ask "Qwen 3.5 4B"                 — то же для локальной модели (нужен запущенный Ollama)
#   Claude: скрипт пишет tandem/claude_prompts_1.txt, _2.txt ... (по 20 алертов); ответы Claude (по строке JSON
#           с полем "i") дописывать в tandem/claude_answers.txt и запустить скрипт ещё раз.
import json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import OLD, OUT, THETA, utf8
from src.llm import EXAMPLE, LLMS, SYSTEM, ask, parse, prompt

T = OUT / "tandem"
REUSE_OLD = False   # True — брать готовые объяснения старой версии; False — все объяснения запрашиваются у итоговой модели заново
CLAUDE_PART = 20    # сколько алертов в одной части для окна Claude
FIELDS = ["priority", "attack_signs", "what_happened", "why_suspicious", "recommendation"]


def old_bank(fname):
    """Готовые объяснения старой версии: номер алерта в выборке 1 → объяснение. Новые версии важнее старых."""
    bank = {}
    for stage in ("stage_10_final", "stage_13_ensemble"):
        f = OLD / stage / fname
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d["source"] == "ML+LLM") & d["what_happened"].fillna("").astype(str).str.strip().ne("")]
        for r in d.itertuples():
            bank[int(r.i)] = {**{k: getattr(r, k) for k in FIELDS},
                               "llm_s": None if "claude" in fname else round(float(r.latency_s), 2)}
    return bank


def main():
    utf8()
    T.mkdir(parents=True, exist_ok=True)
    best = json.load(open(OUT / "ml_results.json", encoding="utf-8"))["best"]
    p_all = np.load(OUT / "scores" / f"{best}.npy")
    s1 = pd.read_csv(OUT / "sample1.csv")
    proba = p_all[s1["alert_id"].values]
    flag = proba >= THETA
    ctx = pd.read_parquet(OUT / "context.parquet", columns=["same_event_1m", "same_event_5m", "same_event_15m"])
    test_rid = pd.read_parquet(OUT / "alerts.parquet", columns=["rid", "scenario", "part"])
    test_rid = test_rid[test_rid.part == "late"].sort_values(["scenario", "rid"], kind="stable")["rid"].values
    same = np.expm1(ctx.values[test_rid[s1["alert_id"].values]]).round()
    print(f"{best} отобрал {flag.sum()} из 200 алертов выборки 1")

    ask_name = sys.argv[sys.argv.index("--ask") + 1] if "--ask" in sys.argv else None
    summary = []
    for name, (how, model, fname) in LLMS.items():
        bank = old_bank(fname) if REUSE_OLD else {}
        cache_f = T / f"new__{re.sub(r'[^A-Za-z0-9]+', '_', name)}.json"
        new = {int(k): v for k, v in json.load(open(cache_f, encoding="utf-8")).items()} if cache_f.exists() else {}
        if how == "manual" and (T / "claude_answers.txt").exists():
            for line in (T / "claude_answers.txt").read_text(encoding="utf-8").splitlines():
                m = re.search(r"\{.*\}", line)
                if m:
                    d = json.loads(m.group(0))
                    new[int(d["i"])] = {**parse(m.group(0)), "llm_s": None}
        missing = [i for i in np.where(flag)[0] if i not in bank and i not in new]

        if ask_name == name and how != "manual":
            failed = []
            for k, i in enumerate(missing, 1):
                for attempt in range(3):                       # сбой на стороне сервиса — повторяем до 3 раз
                    try:
                        t0 = time.perf_counter()
                        raw = ask(how, model, prompt(s1.iloc[i], float(proba[i]), same[i]))
                        break
                    except Exception as e:
                        print(f"  {name}: алерт {i} — ошибка ({e}), попытка {attempt + 1} из 3", flush=True)
                        time.sleep(5)
                else:
                    failed.append(int(i))
                    continue
                new[int(i)] = {**parse(raw), "llm_s": round(time.perf_counter() - t0, 2)}
                print(f"  {name}: [{k}/{len(missing)}] алерт {i} -> {new[int(i)]['priority']}", flush=True)
                json.dump(new, open(cache_f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            missing = failed
        if how == "manual":
            for f in T.glob("claude_prompts*.txt"):
                f.unlink()
            for k in range(0, len(missing), CLAUDE_PART):
                part = missing[k:k + CLAUDE_PART]
                with open(T / f"claude_prompts_{k // CLAUDE_PART + 1}.txt", "w", encoding="utf-8") as f:
                    f.write("Вставьте в окно Claude целиком. Ответ — по одной строке JSON на алерт, с полем \"i\" "
                            "(номер алерта) и полями priority, attack_signs, what_happened, why_suspicious, recommendation.\n\n"
                            "ИНСТРУКЦИЯ:\n" + SYSTEM + "\n\n" + EXAMPLE)
                    for i in part:
                        f.write(f"=== Алерт i={i} ===\n{prompt(s1.iloc[i], float(proba[i]), same[i], with_example=False)}\n")

        rows = []
        for i in range(len(s1)):
            base = {"i": i, "label": s1.at[i, "label"], "group": s1.at[i, "group"], "ml_proba": round(float(proba[i]), 4)}
            if not flag[i]:
                rows.append({**base, "priority": "low", "source": "ML", "explanation_from": ""})
            elif i in new:
                rows.append({**base, **new[i], "source": "ML+LLM", "explanation_from": "новый запрос"})
            elif i in bank:
                rows.append({**base, **bank[i], "source": "ML+LLM", "explanation_from": "старая версия"})
            else:
                rows.append({**base, "priority": "medium", "source": "ML (нет объяснения)", "explanation_from": ""})
        pd.DataFrame(rows).to_csv(T / f"{name}.csv", index=False, encoding="utf-8-sig")
        summary.append({"модель": name, "отобрано": int(flag.sum()), "из старой версии": sum(i in bank and i not in new for i in np.where(flag)[0]),
                        "новых": sum(i in new for i in np.where(flag)[0]), "не хватает": len(missing)})
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    if s["не хватает"].sum():
        print("\nНе хватает объяснений. Запустите с --ask \"<модель>\" (и для Claude — см. tandem/claude_prompts_1.txt, _2.txt …).")


if __name__ == "__main__":
    main()
