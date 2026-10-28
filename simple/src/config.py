# config.py — все настройки простой версии в одном месте.
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]          # D:\Default Project
SIMPLE = PROJECT / "simple"
OUT = SIMPLE / "results"                               # всё, что создаёт простая версия
RAW = PROJECT / "data" / "raw" / "ait_ads"             # исходный набор AIT-ADS
LABELS_CSV = RAW / "labels.csv"
LABELED = PROJECT / "data" / "processed" / "02_labeled_alerts.csv"   # размеченная таблица (если уже есть)
OLD = PROJECT / "results"                              # результаты старой версии (готовые ответы LLM)

SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]
GROUPS = {"A": ["fox", "harrison", "russellmitchell", "santos"],
          "B": ["shaw", "wardbeck", "wheeler", "wilson"]}

SEED = 42
PER_LABEL = 150000 // 11       # не больше 13 636 алертов каждой метки в обучении
THETA = 0.5                    # порог «атака / не атака», не подбирается
LOG_CHARS = 400                # сколько символов журнала берём в текст

BURST_WINDOWS = (10, 30, 60, 300, 900, 3600)   # секунды: всплески одинаковых событий
ACT_WINDOWS = (10, 60, 300, 3600)              # секунды: общая активность хоста и источника


def utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
