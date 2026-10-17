# 52_export_judge_prompts.py
# Из слепого листа reports/Оценка_объяснений_слепая.xlsx делает тексты для ИИ-судей (ChatGPT, Gemini и др.):
# reports/judge/judge_part_1.txt ... judge_part_6.txt (по 5 алертов). Названия моделей в текст не попадают.
# Ответ судьи сохранить в reports/judge/answers_<имя>_part_N.txt (например answers_gemini_part_1.txt).
import re
from pathlib import Path
import openpyxl
ROOT = Path(__file__).resolve().parents[1]
ws = openpyxl.load_workbook(ROOT / "reports" / "Оценка_объяснений_слепая.xlsx")["Оценка"]
alerts, cur = [], None
for r in ws.iter_rows(min_row=4, values_only=True):
    a, b, c = r[0], r[1], r[2]
    if isinstance(a, str) and a.startswith("Алерт"):
        cur = {"head": a, "v": {}}; alerts.append(cur)
    elif b in ("A", "B", "C") and cur is not None:
        cur["v"][b] = c
HEAD = """Ты — независимый эксперт по информационной безопасности (SOC). Оцени объяснения алертов SIEM.
Для каждого алерта даны три варианта объяснения (A, B, C). Они написаны разными моделями; какими именно — неизвестно, не пытайся угадать.
Поставь каждому варианту три оценки по шкале 0, 1 или 2:
 1) Что произошло — верно ли описано событие по тексту алерта и лога (0 — неверно, 1 — частично, 2 — верно);
 2) Почему подозрительно — понятна и обоснована ли причина, нет ли выдуманных фактов (0 — нет, 1 — частично, 2 — да);
 3) Рекомендация — можно ли по ней действовать аналитику, конкретна ли она (0 — бесполезна, 1 — частично, 2 — да).
Оценивай по содержанию, а не по длине. Не придумывай фактов, которых нет в алерте.
Формат ответа: ровно одна строка на каждый вариант, без пояснений и без таблиц, в виде:
номер_алерта буква оценка1 оценка2 оценка3
Пример: 1 A 2 1 2

"""
out = ROOT / "reports" / "judge"; out.mkdir(exist_ok=True)
n = 6; size = -(-len(alerts) // n)
for k in range(n):
    part = alerts[k * size:(k + 1) * size]
    body = ""
    for al in part:
        num = re.match(r"Алерт (\d+)", al["head"]).group(1)
        body += f"=== {al['head']}\n" + "".join(f"--- Вариант {x} (алерт {num}):\n{al['v'][x]}\n" for x in "ABC") + "\n"
    (out / f"judge_part_{k+1}.txt").write_text(HEAD + body, encoding="utf-8")
    print(k + 1, len(part))
