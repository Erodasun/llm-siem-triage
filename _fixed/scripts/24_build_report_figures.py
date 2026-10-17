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
plt.rcParams["font.family"] = "Times New Roman"
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
ax.set_xlabel("Сценарий"); ax.set_ylabel("Атакующих алертов в тесте")
save(fig, "fig2_test_attacks.png")

# --- Рис. 3 ---
sp = pd.read_csv(f"{R}/stage_2/experiment_sample.csv")
c = collections.Counter(sp["label"])
ls, vs = list(c.keys()), list(c.values())
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(ls, vs, color="#4C72B0")
ax.set_xlabel("Метка"); ax.set_ylabel("Алертов")
ax.tick_params(axis="x", rotation=35)
for i, v in enumerate(vs):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
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

# --- Рис. 5-7: F1 по атакам на выборке 100 (источник — 39_all_models_attack_metrics.py) ---
am = pd.read_csv(f"{R}/stage_8/all_models_attack_metrics.csv")
baseline = float(am.loc[am["family"] == "Базовая линия", "F1"].iloc[0])


def f1_bars(sub, name, color):
    sub = sub.sort_values("F1", ascending=False)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(sub["model"], sub["F1"], color=color)
    for b, f1, found, fp in zip(bars, sub["F1"], sub["found"], sub["fp"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{f1:.2f}\n{found}/{fp}", ha="center", fontsize=7)
    ax.axhline(baseline, color="#C44E52", ls="--", lw=1, label=f"«всё — атака» F1={baseline:.2f}")
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub["model"], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("F1 по классу «атака»"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    save(fig, name)


llm = am[am["family"] == "LLM"]
f1_bars(llm[llm["model"].str.startswith("ollama")], "fig5_local_llm.png", "#4C72B0")
f1_bars(llm[llm["model"].str.startswith("polza")], "fig6_polza.png", "#4C72B0")
f1_bars(llm[llm["model"].str.startswith("tokify")], "fig7_tokify.png", "#55A868")

# --- Рис. 8 ---
sp8 = am[am["family"].isin(["LLM", "ML", "Тандем"]) & am["latency_s"].notna() & (am["latency_s"] > 0)]
sp8 = sp8.sort_values("latency_s")
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(sp8["model"], sp8["latency_s"], color="#C44E52")
ax.set_xticks(range(len(sp8))); ax.set_xticklabels(sp8["model"], rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Секунд на алерт (лог. шкала)"); ax.set_yscale("log")
save(fig, "fig8_speed.png")

# --- Рис. 9 ---
hy = pd.read_csv(f"{R}/stage_5/hybrid_summary.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(hy["theta"], hy["filtered_pct"], "o-", label="Отфильтровано алертов (доля)", color="#4C72B0")
ax.plot(hy["theta"], hy["lost_pct"], "s--", label="Потеряно атак (доля)", color="#C44E52")
ax.set_xlabel("Порог θ (вероятность атаки)"); ax.set_ylabel("Доля потока")
ax.legend()
save(fig, "fig9_hybrid.png")

# --- Рис. 10 ---
exp = pd.read_csv(f"{R}/stage_8/tandem_1.csv")
order = ["low", "medium", "high", "critical"]
vals = [int((exp["priority"] == p).sum()) for p in order]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(order, vals, color=["#55A868", "#4C72B0", "#DD8452", "#C44E52"])
ax.set_ylabel("Количество алертов")
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
save(fig, "fig10_priority.png")


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
    f, ax = newfig(11, 4.2)
    box(ax, 0.10, 0.62, 0.14, 0.16, "События SIEM\n(2,65 млн алертов)", GREY)
    box(ax, 0.285, 0.62, 0.14, 0.16, "Признаки\nи текст алерта")
    box(ax, 0.465, 0.62, 0.14, 0.16, "ML-фильтр\nLightGBM", BLUE, bold=True)
    box(ax, 0.67, 0.62, 0.15, 0.20, "LLM\nQwen 3.5 4B\n(few-shot)", PURPLE, bold=True)
    box(ax, 0.86, 0.62, 0.14, 0.16, "Дашборд\nтриажа (Flask)", BLUE)
    arrow(ax, 0.17, 0.62, 0.215, 0.62)
    arrow(ax, 0.355, 0.62, 0.395, 0.62)
    arrow(ax, 0.60, 0.66, 0.595, 0.70, text="p ≥ 0,5", dy=0.01)
    arrow(ax, 0.745, 0.62, 0.79, 0.62)
    box(ax, 0.465, 0.28, 0.19, 0.14, "ФОН — автоматическое\nзакрытие, запись\nв журнал", GREEN)
    box(ax, 0.86, 0.28, 0.15, 0.14, "Аналитик SOC\n(реагирует\nна эскалации)", ORANGE)
    arrow(ax, 0.465, 0.42, 0.465, 0.35)
    arrow(ax, 0.86, 0.42, 0.86, 0.35)
    save(f, "fig11_hybrid_architecture.png")


# --- Рисунок 12. Пайплайн эксперимента ---
def diag_pipeline():
    f, ax = newfig(9.5, 5.5)
    steps = ["Сбор и нормализация алертов", "Разметка по сценариям (8 имитаторов)",
             "Временнóе разбиение train/test", "Единая выборка 100 алертов",
             "Классические ML-модели", "Локальные LLM (Ollama)", "Облачные LLM (Polza/Tokify)",
             "Тандем ML + LLM", "Веб-дашборд триажа"]
    ys = np.linspace(0.9, 0.08, len(steps))
    for i, (s, y) in enumerate(zip(steps, ys)):
        box(ax, 0.5, y, 0.66, 0.072, f"{i + 1}. {s}", size=10.5)
        if i < len(steps) - 1:
            arrow(ax, 0.5, y - 0.036, 0.5, ys[i + 1] + 0.036)
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
    box(ax, 0.50, 0.55, 0.20, 0.24, "Локальная LLM\nQwen 3.5 4B\nтемпература = 0\nрежим JSON",
        PURPLE, bold=True, size=11)
    box(ax, 0.80, 0.55, 0.24, 0.30,
        "JSON-ответ:\npriority\n• что произошло\n• почему подозрительно\n• как реагировать",
        GREEN, size=10.5)
    arrow(ax, 0.30, 0.55, 0.40, 0.55, text="запрос события")
    arrow(ax, 0.60, 0.55, 0.68, 0.55, text="семантический анализ")
    ax.text(0.20, 0.16, "источник данных SIEM", ha="center", fontsize=10, color="#404040")
    ax.text(0.50, 0.16, "локальный GPU (GTX 1050 Ti),\nданные не покидают периметр",
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
    box(ax, 0.34, 0.47, 0.19, 0.14, "Автоматическое\nзакрытие фона\n(96,5 % потока)", GREEN)
    box(ax, 0.78, 0.47, 0.19, 0.14, "Расследование\nи реагирование\nна инциденты", BLUE)
    arrow(ax, 0.625, 0.68, 0.375, 0.545, "алерт = «фон»")
    arrow(ax, 0.625, 0.68, 0.745, 0.545, "подозрительно / критично")
    box(ax, 0.5, 0.20, 0.22, 0.13, "База знаний,\nуточнение правил", GREY)
    arrow(ax, 0.78, 0.40, 0.55, 0.27, "результаты анализа")
    arrow(ax, 0.45, 0.27, 0.452, 0.68, color="#7F7F7F", style="<|-|>")
    ax.text(0.42, 0.42, "замкнутый контур", rotation=90, fontsize=9, color="#7F7F7F")
    save(f, "fig15_soc_triage_place.png")


diag_arch()
diag_pipeline()
diag_flowchart()
diag_llm()
diag_soc()
print("ALL FIGURES OK")
