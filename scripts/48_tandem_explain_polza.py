# 48_tandem_explain_polza.py
# Облачная версия главного тандема: LightGBM отбирает алерты -> GPT-6-sol (Polza) объясняет.
# Те же отобранные алерты, что и в 47, но объяснения даёт облачная модель. Отдельный файл результатов,
# основной прогон 47 (локальный Qwen) не трогается.
# Запуск: python scripts\48_tandem_explain_polza.py [--model openai/gpt-6-sol] [--reasoning-effort low]
#          [--budget 37] [--limit N] [--out results\stage_7\tandem_main_gpt.csv]
# Возобновляемый: готовые алерты пропускаются. Стоимость берётся из поля cost_rub ответа API.
import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROVIDERS, RESULTS, SAMPLE_CTX, utf8_console

S7 = RESULTS / "stage_7"


def _load47():
    spec = importlib.util.spec_from_file_location("t47", Path(__file__).resolve().parent / "47_tandem_explain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t47 = _load47()


def explain(provider, model, system, user, max_tokens, reasoning_effort):
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"], "")
    if not key:
        raise SystemExit(f"{cfg['key_env']} not set")
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(cfg["api"], data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read().decode("utf-8"))
    content = resp["choices"][0]["message"].get("content") or ""
    return content, resp.get("usage", {})


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-6-sol")
    ap.add_argument("--provider", default="polza")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--budget", type=float, default=37.0, help="стоп, если суммарная стоимость достигла этой суммы, руб")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(S7 / "tandem_main_gpt.csv"))
    a = ap.parse_args()
    out = Path(a.out)

    sample = pd.read_csv(SAMPLE_CTX)
    ml = pd.read_csv(S7 / "ml_only_lightgbm.csv").sort_values("i").reset_index(drop=True)
    ml_lat = json.load(open(S7 / "ml_predict_latency.json"))["LightGBM"]
    flag = ml["priority"].ne("low").values
    print(f"LightGBM отобрал {flag.sum()} из {len(ml)} алертов; объясняет {a.model} (reasoning={a.reasoning_effort}, бюджет {a.budget} руб)")

    rows, done, spent = [], {}, 0.0
    if out.exists():
        prev = pd.read_csv(out)
        ok = ~prev["raw"].fillna("").astype(str).str.startswith("ERROR")
        done = {int(r["i"]): r.to_dict() for _, r in prev[ok].iterrows()}
        spent = float(pd.to_numeric(prev.get("cost_rub", 0), errors="coerce").fillna(0).sum())
        print(f"продолжение: готово {len(done)}, уже потрачено ~{spent:.2f} руб")

    n_llm, stopped = 0, False
    for i in range(len(ml)):
        if i in done:
            rows.append(done[i])
            continue
        base = {"i": i, "label": ml.at[i, "label"], "ml_proba": ml.at[i, "ml_proba"], "llm_model": ""}
        if not flag[i]:
            rows.append({**base, "priority": "low", "source": "ML", "attack_signs": "", "latency_s": round(ml_lat, 6),
                         "cost_rub": 0.0, "what_happened": "", "why_suspicious": "", "recommendation": "", "raw": ""})
            continue
        if a.limit and n_llm >= a.limit:
            continue
        if spent >= a.budget:
            stopped = True
            break
        t0 = time.perf_counter()
        try:
            raw, usage = explain(a.provider, a.model, t47.SYSTEM,
                                 t47.build_prompt(sample.iloc[i], float(ml.at[i, "ml_proba"])),
                                 a.max_tokens, a.reasoning_effort)
        except Exception as e:
            raw, usage = f"ERROR: {e}", {}
        p = t47.parse(raw)
        dt = time.perf_counter() - t0
        cost = float(usage.get("cost_rub", 0) or 0)
        spent += cost
        n_llm += 1
        rows.append({**base, "priority": p["priority"], "source": "ML+LLM", "llm_model": a.model,
                     "attack_signs": p["attack_signs"], "latency_s": round(ml_lat + dt, 3), "cost_rub": round(cost, 5),
                     "what_happened": p["what_happened"], "why_suspicious": p["why_suspicious"],
                     "recommendation": p["recommendation"], "raw": raw})
        print(f"[{n_llm:>3}/{flag.sum()}] {base['label']:<22} -> {p['priority']:<8} признаки атаки: {p['attack_signs'] or '?':<3} "
              f"{dt:.1f}s  cost={cost:.4f}  всего={spent:.2f} руб")
        if n_llm % 10 == 0:
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        if spent >= a.budget:
            stopped = True
            print(f"предохранитель: достигнут бюджет {a.budget} руб, останавливаюсь")
            break

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False, encoding="utf-8-sig")
    llm_rows = d[d["source"] == "ML+LLM"]
    bad = llm_rows["raw"].astype(str).str.startswith("ERROR").sum() + (llm_rows["what_happened"].fillna("") == "").sum()
    print(f"готово: {len(llm_rows)} объяснений, без объяснения/с ошибкой: {bad}; "
          f"LLM не видит признаков атаки: {(llm_rows['attack_signs'] == 'нет').sum()}; "
          f"потрачено ~{spent:.2f} руб; остановлен: {stopped} -> {out}")


if __name__ == "__main__":
    main()
