# 24_build_report_figures.py
# Генерация ВСЕХ рисунков отчёта: графики (fig1-fig10) и схемы/блок-диаграммы (fig11-fig15).
# (объединяет прежние 24_generate_report_figures.py и 25_generate_report_diagrams.py)
import os, collections
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

from src.config import FIGURES, RESULTS

R = str(RESULTS)
FIG = str(FIGURES)
os.makedirs(FIG, exist_ok=True)


def save(fig, name, dpi=150):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(FIG, name), dpi=dpi)
    plt.close(fig)
    print("saved:", name)


# ============================================================
# Часть 1. Графики отчёта (Рисунки 1-10)
# ============================================================
plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11

# --- Рис. 1 ---
df = pd.read_csv(f"{R}/stage_1/split_by_label.csv").set_index("split")
labels = df.columns.tolist()
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.bar(x - 0.2, df.loc["train"].clip(lower=0.8), 0.4, label="Обучающая выборка", color="#4C72B0")
ax.bar(x + 0.2, df.loc["test"].clip(lower=0.8), 0.4, label="Тестовая выборка", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Количество алертов (лог. шкала)")
ax.legend(); ax.set_yscale("log")
save(fig, "fig1_split_train_test.png")

# --- Рис. 2 ---
sc = pd.read_csv(f"{R}/stage_1/split_by_scenario.csv").set_index("scenario")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(sc.index, sc["test_attack"], color="#55A868")
ax.set_xlabel("Среда AIT-ADS"); ax.set_ylabel("Атакующих алертов в тесте")
save(fig, "fig2_test_attacks.png")

# --- Рис. 3 ---
sp = pd.read_csv(f"{R}/stage_2/experiment_sample.csv")
c = collections.Counter(sp["label"])
ls, vs = list(c.keys()), list(c.values())
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
ax.bar(ls, vs, color="#4C72B0")
ax.set_xlabel("Метка"); ax.set_ylabel("Алертов")
ax.tick_params(axis="x", rotation=35)
for i, v in enumerate(vs):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
gn = {"easy_attack": "лёгкие\nатаки", "hard_attack": "трудные\nатаки", "hard_benign": "парный\nфон", "easy_benign": "обычный\nфон"}
gc = sp["group"].value_counts().reindex(list(gn))
ax2.bar([gn[k] for k in gc.index], gc.values, color=["#C0504D", "#DD8452", "#7FA6D6", "#4C72B0"])
for i, v in enumerate(gc.values):
    ax2.text(i, v + 0.8, str(v), ha="center", fontsize=9)
ax2.set_xlabel("Группа"); ax2.tick_params(axis="x", labelsize=8)
save(fig, "fig3_sample100.png")

# --- Рис. 4 ---
ml = pd.read_csv(f"{R}/stage_3/ml_models_comparison.csv")
models = ml["model"].tolist(); x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(x - 0.25, ml["precision"], 0.25, label="Precision", color="#4C72B0")
ax.bar(x, ml["recall"], 0.25, label="Recall", color="#DD8452")
ax.bar(x + 0.25, ml["F1"], 0.25, label="F1", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Значение метрики")
ax.legend()
for i in range(len(models)):
    ax.annotate(f"{ml['F1'][i]:.2f}", (i + 0.25, ml['F1'][i] + 0.01), ha="center", fontsize=8)
save(fig, "fig4_ml_metrics.png")

# --- Графики результатов: все данные берутся из CSV ---
import json as _json

PRETTY = {
    "polza__openai-gpt-6-sol__few__none": "GPT-6-sol (облако)",
    "ollama__qwen3.5-4b-q4_K_M__few__none": "Qwen 3.5 4B",
    "ollama__qwen3.5-4b-q4_K_M__few__freq": "Qwen 3.5 4B + контекст частоты",
    "ollama__llama3.1-latest__few__none": "Llama 3.1 8B",
    "ollama__gemma2-2b__few__none": "Gemma 2 2B",
    "ollama__llama3.2-3b__few__none": "Llama 3.2 3B",
    "claude__claude-opus-5-5-manual__few__none": "Claude Opus 5.5",
    "tandem_1": "ИЛИ-тандем 1 (LightGBM или Qwen+контекст)",
    "tandem_2": "ИЛИ-тандем 2 (LightGBM или GPT-6-sol)",
    "tandem_main__ollama__qwen3.5-4b-q4_K_M": "LightGBM → Qwen 3.5 4B",
    "tandem_main__polza__openai-gpt-6-sol": "LightGBM → GPT-6-sol",
    "tandem_main__claude__manual": "LightGBM → Claude Opus 5.5",
}
FAM_COLOR = {"Базовая линия": "#7F7F7F", "LLM": "#6B3FA0", "ML": "#2E5C8A", "Тандем": "#C0504D", "Эксперимент": "#8A6BB8"}


def pretty(name):
    if name in PRETTY:
        return PRETTY[name]
    n = name.split(" θ")[0]
    return n.replace("[text+bursts]", " (текст + всплески)").replace("[text]", " (только текст)")


am = pd.read_csv(f"{R}/stage_7/all_models_attack_metrics.csv")
am["label"] = am["model"].map(pretty)
# три главных тандема дают одинаковые решения (решает LightGBM): на графиках качества показываем одну строку
am_all = am.copy()
_dup = am["model"].isin(["tandem_main__ollama__qwen3.5-4b-q4_K_M", "tandem_main__claude__manual"])
am = am[~_dup].copy()
am.loc[am["model"] == "tandem_main__polza__openai-gpt-6-sol", "label"] = "Тандем: LightGBM → LLM объясняет (решает ML)"
baseline = float(am.loc[am["family"] == "Базовая линия", "F1"].iloc[0])
b_lo = float(am.loc[am["family"] == "Базовая линия", "F1_lo"].iloc[0])
b_hi = float(am.loc[am["family"] == "Базовая линия", "F1_hi"].iloc[0])


def f1_forest(sub, name, title_h=None):
    sub = sub.sort_values("F1")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(sub) + 1.4))
    y = np.arange(len(sub))
    ax.axvspan(b_lo, b_hi, color="#7F7F7F", alpha=0.15, label=f"базовая линия «всё — атака»: F1 = {baseline:.2f} (95 % ДИ {b_lo:.2f}–{b_hi:.2f})")
    ax.axvline(baseline, color="#7F7F7F", ls="--", lw=1)
    for yi, (_, r) in zip(y, sub.iterrows()):
        c = FAM_COLOR[r["family"]]
        ax.errorbar(r["F1"], yi, xerr=[[r["F1"] - r["F1_lo"]], [r["F1_hi"] - r["F1"]]], fmt="o", color=c, capsize=3, lw=1.6)
        ax.text(r["F1_hi"] + 0.012, yi, f"{r['F1']:.2f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(sub["label"], fontsize=8)
    ax.set_xlim(0.1, 0.95); ax.set_xlabel("F1 по классу «атака» (точка — значение, отрезок — 95 % доверительный интервал)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="x", alpha=0.25)
    save(fig, name)


models_all = am[am["family"] != "Базовая линия"]
f1_forest(am[am["family"].isin(["LLM", "Базовая линия"])], "res_f1_llm.png")
f1_forest(models_all, "res_f1_all.png")

# --- precision–recall ---
fig, ax = plt.subplots(figsize=(8, 5.4))
rr = np.linspace(0.05, 1, 200)
for f in (0.5, 0.6, 0.7, 0.8):
    pp = f * rr / (2 * rr - f)
    ok = pp > 0
    ax.plot(rr[ok & (pp <= 1.0)], pp[ok & (pp <= 1.0)], color="#BBBBBB", lw=0.8, ls=":")
    ax.text(1.005, f / (2 - f), f"F1={f}", fontsize=7, color="#888888", va="center")
for _, r in am.iterrows():
    ax.scatter(r["recall"], r["prec"], color=FAM_COLOR[r["family"]], s=45, zorder=3)
    off = {"ИЛИ-тандем 1": (-6, -20, "right"), "ИЛИ-тандем 2": (6, 8, "left"), "GPT-6-sol": (-6, 6, "right"), "Тандем:": (6, -12, "left")}
    dx, dy, ha = next((v for k, v in off.items() if r["label"].startswith(k)), (4, 3, "left"))
    ax.annotate(r["label"].replace(" (LightGBM или ", "\n(LightGBM или ").replace(" (решает ML)", "\n(решает ML)"), (r["recall"], r["prec"]), fontsize=6.5,
                xytext=(dx, dy), textcoords="offset points", ha=ha)
ax.set_xlabel("Recall (полнота по атакам)"); ax.set_ylabel("Precision (точность по атакам)")
ax.set_xlim(0.15, 1.08); ax.set_ylim(0.35, 1.0); ax.grid(alpha=0.2)
for fam, c in FAM_COLOR.items():
    ax.scatter([], [], color=c, label=fam)
ax.legend(fontsize=8, loc="upper right")
save(fig, "res_pr_scatter.png")

# --- лёгкие / трудные атаки и парный фон ---
g = am.sort_values("F1", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
x = np.arange(len(g)); w = 0.27
ax.bar(x - w, g["recall_easy"], w, label="Recall на лёгких атаках", color="#4C72B0")
ax.bar(x, g["recall_hard"], w, label="Recall на трудных атаках", color="#DD8452")
ax.bar(x + w, g["fp_hard_benign"], w, label="Ложные на парном фоне (тот же текст)", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(g["label"], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("Доля алертов группы"); ax.set_ylim(0, 1.25); ax.legend(fontsize=8, ncol=3, loc="upper center")
save(fig, "res_groups.png")

# --- порог θ: валидация и выборка ---
cv = pd.read_csv(f"{R}/stage_7/ml_theta_curve.csv")
thj = _json.load(open(f"{R}/stage_7/ml_thresholds.json"))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, m in zip(axs, ("LightGBM", "XGBoost", "RandomForest")):
    d = cv[cv["model"] == m].reset_index(drop=True)
    ax.plot(range(len(d)), d["F1_val"], "o-", ms=3, color="#2E5C8A", label="валидация (50/50)")
    ax.plot(range(len(d)), d["F1_sample"], "s--", ms=3, color="#C0504D", label="оценочная выборка")
    ax.axvline(int(np.argmin(abs(d["theta"] - thj[m]))), color="#7F7F7F", ls=":", lw=1)
    ticks = list(range(0, len(d), 4))
    ax.set_xticks(ticks); ax.set_xticklabels([f"{d['theta'][t]:g}" for t in ticks], fontsize=7, rotation=45)
    ax.set_title(f"{m}  (выбран θ={thj[m]:g})", fontsize=9)
    ax.set_xlabel("Порог θ")
axs[0].set_ylabel("F1 по классу «атака»"); axs[0].legend(fontsize=7)
save(fig, "res_theta.png")

# --- ML: выборка 200 против полного теста ---
m8 = pd.read_csv(f"{R}/stage_7/ml_models_comparison.csv")
m8["label"] = m8["model"].map(lambda s: pretty(s))
fig, ax = plt.subplots(figsize=(9.5, 4))
x = np.arange(len(m8)); w = 0.38
b1 = ax.bar(x - w / 2, m8["F1"], w, label="оценочная выборка (200 алертов, доля атак 44 %)", color="#2E5C8A")
b2 = ax.bar(x + w / 2, m8["test_F1"], w, label="полный тест (156 592 алертов, доля атак 6 %)", color="#DD8452")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(m8["label"], rotation=25, ha="right", fontsize=7.5)
ax.set_ylabel("F1 по классу «атака»"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
save(fig, "res_ml_gap.png")

# --- скорость ---
sp8 = am_all[am_all["family"].isin(["LLM", "ML", "Тандем", "Эксперимент"]) & am_all["latency_s"].notna() & (am_all["latency_s"] > 0)].sort_values("latency_s")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(range(len(sp8)), sp8["latency_s"], color=[FAM_COLOR[f] for f in sp8["family"]])
ax.set_xticks(range(len(sp8))); ax.set_xticklabels(sp8["label"], rotation=35, ha="right", fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("Секунд на алерт (логарифмическая шкала)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=FAM_COLOR[f], label=f) for f in ("ML", "LLM", "Тандем", "Эксперимент")], fontsize=8, loc="upper left")
save(fig, "fig8_speed.png")

# --- распределение приоритетов ---
import glob as _glob
rows = []
srcs = [(pretty(Path(f).stem.replace("llm__", "")), f) for f in sorted(_glob.glob(f"{R}/stage_4/llm__*.csv"))]
srcs += [("LightGBM (по вероятности)", f"{R}/stage_7/ml_only_lightgbm.csv")]
srcs += [(PRETTY[t], f"{R}/stage_7/{t}.csv") for t in ("tandem_main__polza__openai-gpt-6-sol", "tandem_main__claude__manual", "tandem_main__ollama__qwen3.5-4b-q4_K_M", "tandem_1", "tandem_2")]
for nm, f in srcs:
    d = pd.read_csv(f)
    rows.append([nm] + [int((d["priority"] == p).sum()) for p in ("low", "medium", "high", "critical", "error")])
pr = pd.DataFrame(rows, columns=["m", "low", "medium", "high", "critical", "error"])
fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(pr) + 1.5))
left = np.zeros(len(pr))
for p, c in zip(("low", "medium", "high", "critical"), ("#55A868", "#4C72B0", "#DD8452", "#C44E52")):
    ax.barh(pr["m"], pr[p], left=left, color=c, label=p)
    for i, v in enumerate(pr[p]):
        if v >= 8:
            ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=7, color="white")
    left += pr[p].values
ax.invert_yaxis(); ax.set_xlabel("Количество алертов (всего 200)"); ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.tick_params(axis="y", labelsize=8)
save(fig, "res_priority.png")

# --- Рис. 9 ---
hy = pd.read_csv(f"{R}/stage_5/hybrid_summary.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(hy["theta"], hy["filtered_pct"], "o-", label="Отфильтровано алертов (доля)", color="#4C72B0")
ax.plot(hy["theta"], hy["lost_pct"], "s--", label="Потеряно атак (доля)", color="#C44E52")
ax.set_xlabel("Порог θ (вероятность атаки)"); ax.set_ylabel("Доля потока"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
save(fig, "fig9_hybrid.png")

# ============================================================
# Часть 2. Схемы и блок-диаграммы (Рисунки 11-15)
# ============================================================
BLUE = ("#DCE6F1", "#2E5C8A"); PURPLE = ("#E4DFEA", "#6B3FA0")
GREEN = ("#E2EFDA", "#548235"); ORANGE = ("#FDEADA", "#C0504D")
GREY = ("#EDEDED", "#7F7F7F"); YELLOW = ("#FFF2CC", "#B8860B")


def newfig(w, h):
    f, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    return f, ax


def box(ax, cx, cy, w, h, text, colors=BLUE, size=11, bold=False, lw=1.4, pad=0.008):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad={pad}",
                                linewidth=lw, edgecolor=colors[1], facecolor=colors[0]))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, linespacing=1.25,
            fontweight="bold" if bold else "normal")


def diamond(ax, cx, cy, w, h, text, colors=YELLOW, size=11):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, facecolor=colors[0], edgecolor=colors[1], linewidth=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, linespacing=1.2)


def arrow(ax, x1, y1, x2, y2, text="", color="#404040", lw=1.6, size=10, dx=0, dy=0.012, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, shrinkA=1, shrinkB=1))
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="bottom",
                fontsize=size, color=color)


# --- Рисунок 11. Архитектура гибридной системы ---
def diag_arch():
    f, ax = newfig(11, 4.6)
    box(ax, 0.09, 0.60, 0.14, 0.16, "Алерт SIEM\n(текст, время,\nхост)", GREY)
    box(ax, 0.34, 0.80, 0.20, 0.15, "ML-решатель LightGBM\nвероятность атаки ≥ θ", BLUE, bold=True)
    box(ax, 0.34, 0.40, 0.20, 0.15, "LLM (Qwen / GPT-6-sol)\nприоритет ≥ заданного\nуровня", PURPLE, bold=True)
    box(ax, 0.60, 0.60, 0.13, 0.15, "Правило\n«ИЛИ»", YELLOW)
    box(ax, 0.80, 0.60, 0.15, 0.17, "Приоритет +\nобъяснение LLM\n(дашборд, Flask)", BLUE)
    box(ax, 0.80, 0.20, 0.15, 0.14, "Аналитик SOC", ORANGE)
    arrow(ax, 0.16, 0.65, 0.24, 0.78)
    arrow(ax, 0.16, 0.55, 0.24, 0.42)
    arrow(ax, 0.44, 0.78, 0.535, 0.66)
    arrow(ax, 0.44, 0.42, 0.535, 0.54)
    arrow(ax, 0.665, 0.60, 0.725, 0.60)
    arrow(ax, 0.80, 0.515, 0.80, 0.27)
    ax.text(0.34, 0.08, "ML и LLM работают независимо; алерт считается атакой, если так решил хотя бы один", ha="center", fontsize=9, color="#404040")
    save(f, "fig11_hybrid_architecture.png")


# --- Рисунок 12. Пайплайн эксперимента ---
def diag_pipeline():
    f, ax = newfig(9.5, 5.5)
    steps = ["Сбор и нормализация алертов", "Разметка по сценариям (8 имитаторов)",
             "Временнóе разбиение train/test", f"Единая выборка {len(sp)} алертов",
             "Классические ML-модели (выбор порога на валидации)", "Локальные LLM (Ollama)", "Облачная LLM (Polza AI)",
             "Метрики с доверительными интервалами и базовой линией", "Тандемы ML + LLM", "Веб-дашборд триажа"]
    ys = np.linspace(0.9, 0.08, len(steps))
    for i, (s, y) in enumerate(zip(steps, ys)):
        box(ax, 0.5, y, 0.66, 0.056, f"{i + 1}. {s}", size=10.5)
        if i < len(steps) - 1:
            arrow(ax, 0.5, y - 0.028, 0.5, ys[i + 1] + 0.028)
    save(f, "fig12_experiment_pipeline.png")


# --- Рисунок 13. Блок-схема алгоритма триажа ---
def diag_flowchart():
    f, ax = newfig(9.5, 5.5)
    box(ax, 0.5, 0.90, 0.42, 0.09, "Алерт SIEM", GREY, bold=True)
    arrow(ax, 0.5, 0.855, 0.5, 0.80)
    diamond(ax, 0.5, 0.70, 0.42, 0.16, "ML: вероятность\nатаки ≥ θ ?")
    arrow(ax, 0.5, 0.62, 0.5, 0.55, "да")
    box(ax, 0.5, 0.46, 0.42, 0.11, "Передача в LLM:\nприоритет и объяснение")
    arrow(ax, 0.29, 0.70, 0.14, 0.70, "нет")
    box(ax, 0.14, 0.70, 0.22, 0.13, "Закрыть как\n«фон»", GREEN)
    diamond(ax, 0.5, 0.28, 0.44, 0.16, "Приоритет\nhigh / critical ?")
    arrow(ax, 0.5, 0.20, 0.5, 0.13)
    box(ax, 0.5, 0.06, 0.40, 0.09, "Эскалация аналитику SOC", ORANGE)
    save(f, "fig13_triage_flowchart.png")


# --- Рисунок 14. Схема формирования объяснения LLM ---
def diag_llm():
    f, ax = newfig(10.5, 3.9)
    box(ax, 0.20, 0.55, 0.20, 0.26,
        "Промпт:\n— текст события\n— примеры few-shot\n(webshell → critical)", GREY, size=11)
    box(ax, 0.50, 0.55, 0.20, 0.24, "LLM (локально через\nOllama или облачный API)\nтемпература = 0\nрежим JSON",
        PURPLE, bold=True, size=11)
    box(ax, 0.80, 0.55, 0.24, 0.30,
        "JSON-ответ:\npriority\n• что произошло\n• почему подозрительно\n• как реагировать",
        GREEN, size=10.5)
    arrow(ax, 0.30, 0.55, 0.395, 0.55)
    arrow(ax, 0.605, 0.55, 0.675, 0.55)
    ax.text(0.20, 0.16, "источник данных SIEM", ha="center", fontsize=10, color="#404040")
    ax.text(0.50, 0.16, "локально — GPU GTX 1050 Ti,\nоблако — по API (данные уходят\nза периметр)",
            ha="center", fontsize=10, color="#404040")
    ax.text(0.80, 0.16, "дашборд триажа →\nаналитик SOC", ha="center", fontsize=10, color="#404040")
    save(f, "fig14_llm_prompt.png")


# --- Рисунок 15. Место триажа в SOC ---
def diag_soc():
    f, ax = newfig(10, 4.6)
    box(ax, 0.105, 0.78, 0.15, 0.17,
        "Источники событий\nМСЭ, IDS/IPS,\nАВЗ, AD/LDAP,\nWeb-серверы", GREY)
    box(ax, 0.285, 0.78, 0.145, 0.17, "Сбор и\nнормализация\nсобытий")
    box(ax, 0.45, 0.78, 0.145, 0.17, "Корреляция\nи корреляционные\nправила")
    box(ax, 0.625, 0.78, 0.16, 0.19, "ПЕРВИЧНЫЙ ТРИАЖ\n(разработанная\nсистема)", ORANGE, bold=True)
    arrow(ax, 0.18, 0.86, 0.212, 0.86)
    arrow(ax, 0.357, 0.86, 0.377, 0.86)
    arrow(ax, 0.522, 0.86, 0.545, 0.86)
    box(ax, 0.34, 0.47, 0.19, 0.14, "Автоматическое\nзакрытие фона", GREEN)
    box(ax, 0.78, 0.47, 0.19, 0.14, "Расследование\nи реагирование\nна инциденты", BLUE)
    arrow(ax, 0.60, 0.68, 0.40, 0.545)
    ax.text(0.43, 0.63, "алерт = «фон»", fontsize=9, color="#404040", ha="right")
    arrow(ax, 0.66, 0.68, 0.76, 0.545)
    ax.text(0.76, 0.63, "подозрительно / критично", fontsize=9, color="#404040", ha="left")
    box(ax, 0.5, 0.20, 0.22, 0.13, "База знаний,\nуточнение правил", GREY)
    arrow(ax, 0.75, 0.40, 0.57, 0.27)
    ax.text(0.74, 0.31, "результаты анализа", fontsize=9, color="#404040", ha="left")
    arrow(ax, 0.45, 0.27, 0.452, 0.68, color="#7F7F7F", style="<|-|>")
    ax.text(0.405, 0.30, "замкнутый контур", rotation=90, fontsize=9, color="#7F7F7F")
    save(f, "fig15_soc_triage_place.png")


diag_arch()
diag_pipeline()
diag_flowchart()
diag_llm()
diag_soc()
print("ALL FIGURES OK")
