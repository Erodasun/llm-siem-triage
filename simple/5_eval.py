# 5_eval.py — ШАГ 5. Итоговые таблицы: ML против LLM, тандем, качество по сетям.
#
# Таблица 1 — пять ML-моделей (из шага 3) и базовая линия.
# Таблица 2 — языковые модели в одиночку на тех же 200 алертах (готовые результаты старой версии, тот же промпт:
#             модель сама решает, атака или нет). Повторно не запускаются.
# Таблица 3 — тандем: итоговая ML-модель отбирает, LLM объясняет. Сколько отобрано, сколько объяснено,
#             сколько ложных тревог LLM честно назвала «нет признаков атаки», время LLM.
# Таблица 4 — качество итоговой модели по сетям (ROC-AUC).
# Таблица 5 — проверка признаков итоговой модели.
# Результат: simple/results/итог.xlsx и итог.json
# Запуск: python simple/5_eval.py
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import OLD, OUT, utf8
from src.llm import LLMS

LLM_ALONE = {"claude__claude-opus-5-5-manual__few__none": "Claude Opus 5.5",
             "polza__openai-gpt-6-sol__few__none": "GPT-6-sol",
             "polza__openai-gpt-oss-120b__few__none": "gpt-oss-120b",
             "ollama__llama3.1-latest__few__none": "Llama 3.1 8B",
             "ollama__qwen3.5-4b-q4_K_M__few__none": "Qwen 3.5 4B",
             "ollama__gemma2-2b__few__none": "Gemma 2 2B",
             "ollama__llama3.2-3b__few__none": "Llama 3.2 3B"}


def main():
    utf8()
    res = json.load(open(OUT / "ml_results.json", encoding="utf-8"))
    best = res["best"]
    t1 = pd.read_csv(OUT / "ml_comparison.csv")

    old = pd.read_csv(OLD / "stage_13_ensemble" / "all_models_attack_metrics_v3.csv")
    llm = old[old["model"].isin(LLM_ALONE)].copy()
    llm["модель"] = llm["model"].map(LLM_ALONE)
    t2 = llm[["модель", "found", "fp", "recall", "prec", "F1", "F1_lo", "F1_hi", "latency_s"]].rename(columns={
        "found": "найдено из 87", "fp": "ложных", "recall": "полнота", "prec": "точность",
        "F1_lo": "F1 от", "F1_hi": "F1 до", "latency_s": "секунд на алерт"}).sort_values("F1", ascending=False)
    t2["секунд на алерт"] = t2["секунд на алерт"].apply(lambda v: "не измерялось" if pd.isna(v) else round(float(v), 1))
    s1 = res["results"][best]["sample1"]
    t2 = pd.concat([pd.DataFrame([{"модель": f"для сравнения: {best} (итоговая ML)", "найдено из 87": s1["found"],
                                   "ложных": s1["fp"], "полнота": s1["recall"], "точность": s1["precision"], "F1": s1["F1"],
                                   "F1 от": s1["F1_low"], "F1 до": s1["F1_high"],
                                   "секунд на алерт": round(res["ms_per_alert_incl_features"] / 1000, 5)}]), t2])

    rows = []
    for name in LLMS:
        d = pd.read_csv(OUT / "tandem" / f"{name}.csv")
        fl = d[d["source"] != "ML"]
        ex = d[d["source"] == "ML+LLM"]
        fa = ex[ex["label"] == "benign"]
        ta = ex[ex["label"] != "benign"]
        rows.append({"LLM в тандеме": name, "отобрано ML": len(fl), "с объяснением": len(ex),
                     "ложных тревог среди отобранных": int((fl["label"] == "benign").sum()),
                     "из них LLM: «нет признаков атаки»": int((fa["attack_signs"] == "нет").sum()),
                     "настоящих атак, где LLM: «нет признаков»": int((ta["attack_signs"] == "нет").sum()),
                     "медиана секунд LLM на алерт": round(float(ex["llm_s"].median()), 1) if ex["llm_s"].notna().any() else "не измерялось"})
    t3 = pd.DataFrame(rows)
    t4 = pd.read_csv(OUT / "by_scenario.csv")
    t5 = pd.read_csv(OUT / "ablation.csv")

    with pd.ExcelWriter(OUT / "итог.xlsx") as w:
        t1.to_excel(w, sheet_name="1 ML-модели", index=False)
        t2.to_excel(w, sheet_name="2 LLM в одиночку", index=False)
        t3.to_excel(w, sheet_name="3 Тандем", index=False)
        t4.to_excel(w, sheet_name="4 По сетям", index=False)
        t5.to_excel(w, sheet_name="5 Признаки", index=False)
    json.dump({"итоговая модель": best, "ML": t1.to_dict("records"), "LLM": t2.to_dict("records"),
               "тандем": t3.to_dict("records"), "по сетям": t4.to_dict("records"), "признаки": t5.to_dict("records")},
              open(OUT / "итог.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for t in (t1, t2, t3, t4, t5):
        print(t.to_string(index=False), "\n")


if __name__ == "__main__":
    main()
