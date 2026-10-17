# 17_stage6_flask_app.py
# Этап 6: Flask-дашборд триажа алертов (мультимодельный).
#  - выбор модели: локальная Qwen 3.5 4B + облачные флагманы Polza/Tokify
#  - KPI, график распределения, «Итоги», топ критичных, очередь, сравнение моделей
# Использование:
#   --static-only            сгенерировать dashboard*.html и выйти
#   (без аргументов)         запустить сервер http://0.0.0.0:5001,
#                            параметр ?model=<key> переключает модель
import os, sys, io, json, base64, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, render_template_string, request

from src.config import RESULTS

BASE = str(RESULTS)
S6 = os.path.join(BASE, "stage_6")
S7 = os.path.join(BASE, "stage_7")
S8 = os.path.join(BASE, "stage_8")
S4 = os.path.join(BASE, "stage_4")
META = os.path.join(BASE, "stage_2", "experiment_sample.csv")
OUT_HTML = os.path.join(S6, "dashboard.html")

CRIT_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
DEFAULT_MODEL = "tandem_1"

PROVIDER = {"ollama": ("локальная", "local"), "polza": ("Polza AI", "cloud"), "tokify": ("Tokify", "cloud")}


def llm_models():
    rows = []
    for f in sorted(glob.glob(os.path.join(S4, "llm__*.csv"))):
        _, backend, model, shots, ctx = os.path.basename(f)[:-4].split("__")
        prov, kind = PROVIDER.get(backend, (backend, "cloud"))
        title = f"{model} ({shots}-shot{', ' + ctx if ctx != 'none' else ''})"
        rows.append((f"llm_{backend}_{model}_{shots}_{ctx}".replace(".", "_"), title, prov, f, kind, "LLM",
                     f"LLM: {model}, {prov}, {shots}-shot, контекст: {ctx}"))
    return rows


MODELS = [
    ("tandem_1", "Тандем 1: LightGBM + Qwen (частотный контекст)", "Тандем", os.path.join(S8, "tandem_1.csv"), "local", "Тандемы",
     "ML: LightGBM (текст + всплески + контекст) · LLM: Qwen 3.5 4B + частотный контекст. Правило: LightGBM≥θ ИЛИ LLM≥high."),
    ("tandem_2", "Тандем 2: LightGBM + GPT-6-sol", "Тандем", os.path.join(S8, "tandem_2.csv"), "cloud", "Тандемы",
     "ML: LightGBM · LLM: GPT-6-sol (Polza AI). Правило: LightGBM≥θ ИЛИ LLM≥medium."),
    ("ml_decides", "ML решает (LightGBM) → LLM объясняет", "Тандем", os.path.join(S8, "ml_decides_llm_explains.csv"), "local", "Тандемы",
     "ML: LightGBM принимает решение · LLM: Qwen 3.5 4B формирует объяснения только для эскалированных алертов."),
    ("ml_xgboost", "XGBoost (ML-only)", "ML", os.path.join(S8, "ml_only_xgboost.csv"), "local", "ML",
     "Алгоритм: XGBoost (текст + всплески + контекст). LLM не используется, объяснений нет."),
    ("ml_lightgbm", "LightGBM (ML-only)", "ML", os.path.join(S8, "ml_only_lightgbm.csv"), "local", "ML",
     "Алгоритм: LightGBM (текст + всплески + контекст). LLM не используется, объяснений нет."),
    ("ml_randomforest", "RandomForest (ML-only)", "ML", os.path.join(S8, "ml_only_randomforest.csv"), "local", "ML",
     "Алгоритм: RandomForest (тот же набор признаков). LLM не используется, объяснений нет."),
    ("ml_logreg", "LogisticRegression (ML-only)", "ML", os.path.join(S8, "ml_only_logreg.csv"), "local", "ML",
     "Алгоритм: LogisticRegression (тот же набор признаков). LLM не используется, объяснений нет."),
    ("ml_linearsvm", "LinearSVM (ML-only)", "ML", os.path.join(S8, "ml_only_linearsvm.csv"), "local", "ML",
     "Алгоритм: LinearSVM (тот же набор признаков). LLM не используется, объяснений нет."),
]
MODELS = [m for m in MODELS if os.path.exists(m[3])] + llm_models()
GROUP_ORDER = ["Тандемы", "ML", "LLM"]
GROUP_INFO = {
    "Тандемы": ("Гибридные тандемы «ML + LLM»",
                "ML детерминированно и быстро решает, эскалировать ли алерт; LLM (Qwen 3.5 4B) "
                "формирует объяснение и рекомендацию. Здесь сравниваются несколько стратегий объединения: "
                "логическое ИЛИ, каскад, взвешенная сумма, «ML решает — LLM объясняет»."),
    "ML": ("Классические ML-модели (без LLM)",
           "Пять алгоритмов на единых признаках (текст + временные всплески + контекст). Работают мгновенно "
           "и детерминированно, но не дают текстового объяснения — это и есть мотивация гибрида."),
    "LLM": ("Языковые модели (локальные и облачные)",
            "Локальная Qwen 3.5 4B (Ollama) и облачные флагманы Polza AI и Tokify. Дают объяснение и рекомендацию "
            "на русском языке, но медленнее и упираются в текстовый потолок точности на части атак."),
}

HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>SOC Triage Dashboard</title>
<style>
 body{font-family:Segoe UI,Arial;margin:24px;background:#0f172a;color:#e2e8f0}
 h1{color:#38bdf8} h2{font-size:18px;color:#94a3b8;margin-top:28px;border-bottom:1px solid #334155;padding-bottom:4px}
 .nav{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0;padding:10px;background:#1e293b;border-radius:10px}
 .nav a{text-decoration:none;color:#e2e8f0;background:#334155;border-radius:14px;padding:5px 12px;font-size:13px}
 .nav a.on{background:#38bdf8;color:#0f172a;font-weight:600}
 .nav span{color:#64748b;font-size:13px;align-self:center}
 .nav span.grp{color:#38bdf8;font-weight:700;font-size:13px;align-self:center;margin-left:6px}
 tr.grp-row td{background:#0b1220;color:#38bdf8;font-weight:700;letter-spacing:.4px}
 .cards{display:flex;gap:16px;margin:16px 0;flex-wrap:wrap}
 .card{background:#1e293b;border-radius:10px;padding:14px 20px;min-width:150px}
 .card b{font-size:26px;display:block}
 .card small{color:#94a3b8}
 .crit{color:#f87171}.high{color:#fb923c}.med{color:#facc15}.low{color:#4ade80}
 table{border-collapse:collapse;width:100%;background:#1e293b;border-radius:10px;margin-top:8px}
 th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #334155;font-size:13px}
 th{color:#94a3b8}
 .badge{display:inline-block;border-radius:12px;padding:2px 10px;font-weight:600}
 .b-low{background:#14532d;color:#4ade80}.b-medium{background:#713f12;color:#facc15}
 .b-high{background:#7c2d12;color:#fb923c}.b-critical{background:#7f1d1d;color:#f87171}
 .muted{color:#94a3b8;font-size:12px}
 .summary{background:#1e293b;border-radius:10px;padding:14px 20px;margin:12px 0}
 .summary ul{margin:6px 0;padding-left:20px;line-height:1.7}
 .err{color:#f87171}
 .tabs{display:flex;gap:8px;margin:10px 0 0}
 .tabs a{text-decoration:none;font-size:13px;padding:7px 16px;border-radius:8px 8px 0 0;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-bottom:none}
 .tabs a.onx{background:#334155;color:#38bdf8;font-weight:700}
 .comps{background:#0b1220;border-left:3px solid #38bdf8;border-radius:6px;padding:8px 14px;font-size:13px;color:#cbd5e1;margin:10px 0}
 .comps b{color:#38bdf8}
 .note{color:#94a3b8;font-size:13px;font-style:italic}
 a.link{color:#38bdf8;text-decoration:none;font-size:13px}
</style></head><body>
<h1>SOC Triage Dashboard | Триаж алертов SIEM</h1>
<div class="tabs">
 <a href="dashboard.html" class="{% if page_type == 'index' %}onx{% endif %}">Обзор моделей</a>
 {% for grp, href, active in group_tabs %}<a href="{{href}}" class="{% if active %}onx{% endif %}">{{grp}}</a>{% endfor %}
</div>
<div class="nav">
 <span class="grp">{{cur_group}}:</span>
 {% for k, name, href, on in model_links %}{% if on %}<a class="on" href="{{href}}">{{name}}</a>{% else %}<a href="{{href}}">{{name}}</a>{% endif %}{% endfor %}
</div>
<p class="comps"><b>Состав модели:</b> {{components}}</p>
<div class="cards">
  <div class="card"><span class="muted">Всего алертов</span><b>{{total}}</b></div>
  <div class="card"><span class="muted">Авто-закрыто (low)</span><b class="low">{{auto_fixed}}</b></div>
  <div class="card"><span class="muted">Нужна эскалация</span><b class="med">{{escalated}}</b></div>
  <div class="card"><span class="muted">Из них критичные</span><b class="crit">{{critical}}</b></div>
  <div class="card"><span class="muted">F1 по атакам</span><b class="med">{{f1_pct}}%</b></div>
  <div class="card"><span class="muted">Среднее время</span><b>{{lat_avg}}</b></div>
</div>
<h2>Распределение предсказанной критичности — {{model_name}}</h2>
<img src="data:image/png;base64,{{plot}}" width="640" style="background:#0f172a">
<h2>Итоги</h2>
<div class="summary">
 <ul>
  <li>Модель {{model_name}} ({{provider}}), {{total}} алертов единой оценочной выборки.</li>
  <li>Автоматически закрыто как фоновые <b class="low">{{auto_fixed}}</b> алертов; на эскалацию направлено <b class="med">{{escalated}}</b>.</li>
  <li>Приоритет отличный от low интерпретирован как «атака»: совпадение с фактической меткой —
      <b class="med">{{acc_pct}}%</b>. Из {{attacks}} атакующих выявлено <b>{{detected}}</b>
      (полнота {{recall_pct}}%), фоновых ошибочно эскалировано {{fp}} ({{fp_pct}}%).</li>
  <li>Выделено критичных алертов: <b class="crit">{{critical}}</b>.{% if has_llm %} По каждому дано объяснение и рекомендация (LLM).{% else %} Модель не использует LLM, поэтому текстовых объяснений нет — приоритет присвоен по вероятности алгоритма.{% endif %}</li>
  <li>Критичные сценарии выявлены: {{critical_scen}}.{% if crit_benign %} Ложных критичных (фоновые): {{crit_benign}}.{% endif %}</li>
  <li>Пропущены (метка «атака», приоритет low): {{missed_list}}.</li>
 </ul>
</div>
<h2>Топ критичных и высокоприоритетных алертов</h2>
{% if has_llm %}
<table>
<tr><th>Тип</th><th>Приоритет</th><th>Событие</th><th>Что произошло (LLM)</th><th>Как реагировать (LLM)</th></tr>
{% for a in top %}
<tr>
 <td>{{a.true}}</td>
 <td><span class="badge b-{{a.prio}}">{{a.prio}}</span>{% if a.src %}<div class="muted" style="font-size:10px">{{a.src}}</div>{% endif %}</td>
 <td class="muted">{{a.desc}}</td>
 <td>{{a.what}}</td>
 <td>{{a.rec}}</td>
</tr>
{% else %}
<tr><td colspan="5" class="err">Модель не выделила критичных/high алертов.</td></tr>
{% endfor %}
</table>
{% else %}
<p class="note">Чистая ML-модель не формирует текстовых объяснений — приоритет присвоен по вероятности алгоритма. Текстовые объяснения появляются в тандеме (раздел «Тандемы»).</p>
<table>
<tr><th>Тип</th><th>Приоритет</th><th>Событие</th></tr>
{% for a in top %}
<tr><td>{{a.true}}</td><td><span class="badge b-{{a.prio}}">{{a.prio}}</span></td><td class="muted">{{a.desc}}</td></tr>
{% else %}
<tr><td colspan="3" class="err">Модель не выделила критичных/high алертов.</td></tr>
{% endfor %}
</table>
{% endif %}
<h2>Сравнение моделей раздела «{{cur_group}}» &nbsp;<a class="link" href="dashboard.html">сравнить все разделы →</a></h2>
<img src="data:image/png;base64,{{compare_plot}}" width="860" style="background:#0f172a">
<table>
<tr><th>Модель</th><th>Провайдер</th><th>F1 атак</th><th>Critical</th><th>High</th><th>Medium</th><th>Low</th><th>Скорость</th></tr>
{% for m in compare_rows %}
<tr>
 <td>{{m.name}}</td><td>{{m.provider}}</td><td><b>{{m.acc}}%</b></td>
 <td class="crit">{{m.crit}}</td><td class="high">{{m.high}}</td><td class="med">{{m.med}}</td><td class="low">{{m.low}}</td>
 <td>{{m.lat}}</td>
</tr>
{% endfor %}
</table>
<h2>Очередь алертов (по критичности)</h2>
<table>
{% if has_llm %}
<tr><th>#</th><th>Тип</th><th>Критичность</th><th>Промпт (фрагмент)</th><th>Что произошло (LLM)</th><th>Почему подозрительно (LLM)</th><th>Как реагировать (LLM)</th></tr>
{% for a in alerts %}
<tr>
 <td>{{loop.index}}</td>
 <td>{{a.true}}</td>
 <td><span class="badge b-{{a.prio}}">{{a.prio}}</span>{% if a.src %}<div class="muted" style="font-size:10px">{{a.src}}</div>{% endif %}</td>
 <td class="muted">{{a.desc}}</td>
 <td>{{a.what}}</td>
 <td>{{a.why}}</td>
 <td>{{a.rec}}</td>
</tr>
{% endfor %}
{% else %}
<tr><th>#</th><th>Тип</th><th>Критичность</th><th>Промпт (фрагмент)</th></tr>
{% for a in alerts %}
<tr>
 <td>{{loop.index}}</td>
 <td>{{a.true}}</td>
 <td><span class="badge b-{{a.prio}}">{{a.prio}}</span></td>
 <td class="muted">{{a.desc}}</td>
</tr>
{% endfor %}
{% endif %}
</table>
</body></html>
"""


INDEX_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>SOC Triage Dashboard — обзор</title>
<style>
 body{font-family:Segoe UI,Arial;margin:24px;background:#0f172a;color:#e2e8f0}
 h1{color:#38bdf8}
 h2{font-size:19px;color:#38bdf8;margin-top:34px;border-bottom:1px solid #334155;padding-bottom:4px}
 .tabs{display:flex;gap:8px;margin:10px 0 0}
 .tabs a{text-decoration:none;font-size:13px;padding:7px 16px;border-radius:8px 8px 0 0;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-bottom:none}
 .tabs a:hover{color:#38bdf8}
 .intro{background:#1e293b;border-radius:10px;padding:14px 20px;margin:14px 0;line-height:1.6;font-size:14px}
 .intro b{color:#38bdf8}
 .desc{color:#94a3b8;font-size:13px;margin:6px 0 4px}
 table{border-collapse:collapse;width:100%;background:#1e293b;border-radius:10px;margin-top:8px}
 th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #334155;font-size:13px}
 th{color:#94a3b8}
 .crit{color:#f87171}.high{color:#fb923c}.med{color:#facc15}.low{color:#4ade80}
 .muted{color:#94a3b8;font-size:12px;display:block;margin-top:2px}
 a.m{color:#e2e8f0;text-decoration:none}a.m:hover{color:#38bdf8}
 .badge{display:inline-block;border-radius:12px;padding:1px 8px;font-size:11px;font-weight:600;background:#334155;color:#94a3b8}
</style></head><body>
<h1>SOC Triage Dashboard | Обзор моделей</h1>
<div class="tabs">
 <a class="on" style="background:#334155;color:#38bdf8;font-weight:700">Обзор моделей</a>
 {% for grp, href, active in group_tabs %}<a href="{{href}}">{{grp}}</a>{% endfor %}
</div>
<div class="intro">
 Приоритизация алертов SIEM тремя классами решений на единой оценочной выборке
 (атаки по сути + фон, см. 08_prepare_experiment_sample.py). Метрика: F1 по классу «атака» (приоритет medium/high/critical = атака); ориентир — модель «всё — атака». Разделы ниже:
 <b>Тандемы</b> (ML + LLM), <b>ML</b> (без LLM) и <b>LLM</b>.
 <br><b>Объяснения (что произошло / почему подозрительно / как реагировать)</b> формирует LLM, поэтому они есть
 в разделах «Тандемы» и «LLM». Чистые ML-модели объяснений не дают — это и есть довод в пользу гибрида.
 Кликните модель, чтобы открыть её страницу триажа.
</div>
{% for s in sections %}
<h2 id="{{s.id}}">{{s.title}}</h2>
<p class="desc">{{s.desc}}</p>
<table>
<tr><th>Модель</th><th>Провайдер</th><th>F1 атак</th><th>Critical</th><th>High</th><th>Medium</th><th>Low</th><th>Скорость</th></tr>
{% for m in s.rows %}
<tr>
 <td><a class="m" href="{{m.href}}">{{m.name}}</a><span class="muted">{{m.comps}}</span></td>
 <td>{{m.provider}}</td><td><b>{{m.acc}}%</b></td>
 <td class="crit">{{m.crit}}</td><td class="high">{{m.high}}</td><td class="med">{{m.med}}</td><td class="low">{{m.low}}</td>
 <td>{{m.lat}}</td>
</tr>
{% endfor %}
</table>
{% endfor %}
</body></html>
"""


def load_model(key):
    meta, name, provider, path, kind, group, comps = next(m for m in MODELS if m[0] == key)
    df = pd.read_csv(path)
    df["what"] = df["what_happened"].fillna("").astype(str) if "what_happened" in df else ""
    df["why"] = df["why_suspicious"].fillna("").astype(str) if "why_suspicious" in df else ""
    df["rec"] = df["recommendation"].fillna("").astype(str) if "recommendation" in df else ""
    df["lat"] = pd.to_numeric(df["latency_s"], errors="coerce") if "latency_s" in df else np.nan
    df["prio"] = df["priority"].fillna("low").astype(str).str.lower().str.strip()
    df["prio"] = df["prio"].where(df["prio"].isin(CRIT_ORDER), "low")
    if "i" in df:
        df = df.sort_values("i")
    df = df.reset_index(drop=True)
    me = pd.read_csv(META)[["label", "description"]].reset_index(drop=True)
    df["true_label"] = me["label"].values
    df["description"] = me["description"].values
    return df, name, provider


def make_plot(df, title):
    counts = df["prio"].value_counts().to_dict()
    labels = ["critical", "high", "medium", "low"]
    vals = [counts.get(x, 0) for x in labels]
    colors = ["#f87171", "#fb923c", "#facc15", "#4ade80"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.bar(labels, vals, color=colors)
    ax.set_title(title, color="#e2e8f0")
    ax.set_ylabel("alerts")
    ax.tick_params(colors="#94a3b8")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", color="#e2e8f0")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_compare_plot(rows):
    rows = sorted(rows, key=lambda r: r["acc"], reverse=True)
    names = [r["name"] for r in rows]
    accs = [r["acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    bars = ax.bar(names, accs, color="#38bdf8")
    best = max(range(len(accs)), key=lambda i: accs[i])
    bars[best].set_color("#4ade80")
    ax.axhline(accs[best], color="#4ade80", ls="--", lw=1)
    ax.set_title("F1 по классу «атака» (приоритет medium/high/critical = «атака»)", color="#e2e8f0")
    ax.set_ylabel("F1, %")
    ax.set_ylim(0, max(accs) * 1.35 + 5)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", color="#e2e8f0", fontsize=9)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fmt_lat(s):
    if s is None or s != s:
        return "—"
    if s >= 1:
        return f"{s:.2f} с"
    if s >= 0.001:
        return f"{s * 1000:.2f} мс"
    return f"{s * 1e6:.1f} мкс"


def compute_metrics(df):
    true_attack = df["true_label"].ne("benign")
    pred_attack = df["prio"].isin(["medium", "high", "critical"])
    acc_pct = round(100 * (pred_attack == true_attack).mean(), 1)
    det = int((pred_attack & true_attack).sum())
    attacks = int(true_attack.sum())
    fp = int((pred_attack & (~true_attack)).sum())
    recall_pct = round(100 * det / max(1, attacks), 1)
    fp_pct = round(100 * fp / max(1, int((~true_attack).sum())), 1)
    prec = det / max(1, det + fp)
    rec = det / max(1, attacks)
    f1_pct = round(100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0), 1)
    counts = df["prio"].value_counts()
    crit_mask = df["prio"] == "critical"
    crit_attacks = sorted(df.loc[crit_mask & df["true_label"].ne("benign"), "true_label"].unique().tolist())
    crit_benign = int((crit_mask & df["true_label"].eq("benign")).sum())
    critical_scen = ", ".join(crit_attacks) or "нет"
    missed = df[df["true_label"].ne("benign") & df["prio"].eq("low")]
    missed_list = ", ".join(missed["true_label"].value_counts().index.tolist()[:5]) or "нет"
    return dict(acc_pct=acc_pct, f1_pct=f1_pct, detected=det, attacks=attacks, recall_pct=recall_pct,
                fp=fp, fp_pct=fp_pct, critical_scen=critical_scen, crit_benign=crit_benign,
                missed_list=missed_list,
                auto=int(counts.get("low", 0)), escalated=int((df["prio"] != "low").sum()),
                crit=int(counts.get("critical", 0)), high=int(counts.get("high", 0)),
                med=int(counts.get("medium", 0)),
                lat_avg=fmt_lat(float(df["lat"].mean())) if df["lat"].notna().any() else "—")


GSLUG = {"Тандемы": "tandems", "ML": "ml", "LLM": "llm"}


def _group_rows(group):
    rows = []
    for k, n, p, _p, _kind, g, comps in MODELS:
        if g != group:
            continue
        d, _n, _p = load_model(k)
        c = compute_metrics(d)
        rows.append(dict(key=k, name=n, provider=p, group=g, comps=comps,
                         acc=c["f1_pct"], crit=c["crit"], high=c["high"],
                         med=c["med"], low=c["auto"], lat=c["lat_avg"]))
    rows.sort(key=lambda r: -r["acc"])
    return rows


def build_page(key, live=False, queue_limit=None):
    df, name, provider = load_model(key)
    comps = next(m[6] for m in MODELS if m[0] == key)
    group = next(m[5] for m in MODELS if m[0] == key)
    has_llm = group in ("LLM", "Тандемы")
    m = compute_metrics(df)
    has_src = "llm_priority" in df.columns

    def _src(r):
        s = str(r.get("source", "")).strip()
        return {"ML": "решение: ML-модель", "LLM": "решение: LLM", "ML+LLM": "решение: ML+LLM"}.get(s, "")

    def _why(r):
        if has_src and str(r.get("llm_priority", "")).lower() == "low":
            return ("Эскалацию выполнила ML-модель (LightGBM, p≈" + str(r.get("ml_proba", "?")) +
                    "); проверка LLM не подтвердила атаку (priority=low).")
        return str(r["why"])[:170] or "(нет объяснения)"

    top = df[df["prio"].isin(["critical", "high"])].sort_values("prio", key=lambda s: s.map(CRIT_ORDER)).head(8)
    top_rows = [{
        "true": str(r["true_label"])[:22], "prio": r["prio"],
        "desc": str(r["description"])[:50],
        "what": str(r["what"])[:130] or "(нет)", "rec": str(r["rec"])[:150] or "(нет)",
        "src": _src(r),
    } for _, r in top.iterrows()]

    alerts = []
    dfq = df.sort_values("prio", key=lambda s: s.map(CRIT_ORDER))
    for _, r in dfq.iterrows():
        alerts.append({
            "true": str(r["true_label"])[:22], "prio": r["prio"],
            "desc": str(r["description"])[:60],
            "what": str(r["what"])[:150] or "(нет объяснения)",
            "why": _why(r),
            "rec": str(r["rec"])[:170] or "(нет рекомендации)",
            "src": _src(r),
        })
        if queue_limit and len(alerts) >= queue_limit:
            break

    # навигация: только модели ТЕКУЩЕГО раздела
    model_links = []
    for k, n, p, _p, _k, g, _c in MODELS:
        if g != group:
            continue
        href = f"/?model={k}" if live else f"dashboard_{k}.html"
        model_links.append((k, n, href, k == key))

    # вкладки разделов -> якоря на странице обзора
    idxbase = "/" if live else "dashboard.html"
    group_tabs = [(grp, f"{idxbase}#{GSLUG[grp]}", grp == group) for grp in GROUP_ORDER]

    plot = make_plot(df, f"{name} — распределение критичности")

    # сравнение — только внутри текущего раздела
    cmp = _group_rows(group)
    compare_plot = make_compare_plot(cmp)

    app = Flask(__name__)
    with app.app_context():
        return render_template_string(
            HTML, page_type="model", cur_group=group, components=comps, has_llm=has_llm,
            group_tabs=group_tabs, model_links=model_links, model_name=name, provider=provider,
            total=len(df), auto_fixed=m["auto"], escalated=m["escalated"],
            critical=m["crit"], acc_pct=m["acc_pct"], f1_pct=m["f1_pct"], lat_avg=m["lat_avg"],
            plot=plot, attacks=m["attacks"], detected=m["detected"],
            recall_pct=m["recall_pct"], fp=m["fp"], fp_pct=m["fp_pct"],
            critical_scen=m["critical_scen"], crit_benign=m["crit_benign"], missed_list=m["missed_list"],
            top=top_rows, compare_rows=cmp, compare_plot=compare_plot,
            alerts=alerts)


def build_index(live=False):
    sections = []
    for grp in GROUP_ORDER:
        rows = _group_rows(grp)
        for r in rows:
            r["href"] = (f"/?model={r['key']}" if live else f"dashboard_{r['key']}.html")
        title, desc = GROUP_INFO[grp]
        sections.append(dict(id=GSLUG[grp], title=title, desc=desc, rows=rows))
    group_tabs = [(grp, f"#{GSLUG[grp]}", False) for grp in GROUP_ORDER]
    app = Flask(__name__)
    with app.app_context():
        return render_template_string(INDEX_HTML, page_type="index", group_tabs=group_tabs, sections=sections)


def main():
    os.makedirs(S6, exist_ok=True)
    if "--static-only" in sys.argv:
        for old in glob.glob(os.path.join(S6, "dashboard_*.html")):
            os.remove(old)
        idx = build_index()
        with open(os.path.join(S6, "dashboard.html"), "w", encoding="utf-8") as f:
            f.write(idx)
        print("written:", "dashboard.html", len(idx))
        for key, name, prov, path, kind, group, comps in MODELS:
            fn = f"dashboard_{key}.html"
            html = build_page(key, live=False)
            with open(os.path.join(S6, fn), "w", encoding="utf-8") as f:
                f.write(html)
            print("written:", fn, len(html))
        print("static-only: done.")
        return

    app = Flask(__name__)

    @app.route("/")
    def index():
        key = request.args.get("model")
        if not key:
            return build_index(live=True)
        keys = [m[0] for m in MODELS]
        if key not in keys:
            key = DEFAULT_MODEL
        return build_page(key, live=True)

    import socket
    lan = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")]
    print("Launching at http://127.0.0.1:5001 — open in browser.")
    print("Доступно по сети: " + "; ".join("http://" + ip + ":5001" for ip in lan))
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main()