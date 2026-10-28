# features.py — как алерт превращается в числа для модели.
# Четыре группы признаков:
#   text      — слова и кусочки слов из текста алерта (даты скрыты);
#   bursts    — сколько таких же событий пришло недавно в этой сети;
#   activity  — сколько вообще алертов пришло недавно в этой сети из того же источника (Wazuh или AMiner);
#   source    — Wazuh или AMiner.
# Отдельная группа для проверки (в итоговую модель не входит):
#   identity  — название сети (сценарий); в новой организации его не будет.
# Поле host не используется: в AIT-ADS у всех алертов Wazuh хост один и тот же («wazuh-client») во всех сетях,
# у AMiner хоста нет, то есть хост повторяет источник. Счёт «по хосту» без сети к тому же смешивает разные сети,
# которые шли одновременно (например, harrison и wilson), — поэтому такие признаки убраны.
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ACT_WINDOWS, BURST_WINDOWS, LOG_CHARS, SCENARIOS

MAIN_GROUPS = ("text", "bursts", "activity", "source")

# --- текст ---------------------------------------------------------------------------------
MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
TS = re.compile("|".join([
    r"\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}(?:\.\d+)?",                          # Suricata
    MON + r" +\d{1,2} \d{2}:\d{2}:\d{2}",                                        # syslog
    r"\d{1,2}/" + MON + r"/\d{4}:\d{2}:\d{2}:\d{2}(?: [+-]\d{4})?",              # Apache
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",  # ISO
    r"\b1[5-7]\d{8}(?:\.\d+)?",                                                  # время Unix
    r"\b\d{2}:\d{2}:\d{2}\b", r"\b20[12]\d\b", MON + r"\b"]))


IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def text_of(df, mask_ip=False):
    """Название события + начало журнала; даты и время заменены словом TS (по желанию IP-адреса — словом IP)."""
    t = df["description"].fillna("").astype(str) + " " + df["full_log"].fillna("").astype(str).str[:LOG_CHARS]
    t = t.str.replace(TS, " TS ", regex=True)
    return t.str.replace(IP, " IP ", regex=True) if mask_ip else t


# --- всплески и активность (считаются один раз для всех алертов, только по прошлому) ---------
def _window_counts(meta, keys, windows):
    """Для каждого алерта: сколько алертов с тем же ключом пришло за последние w секунд (включая его самого)."""
    out = np.zeros((len(meta), len(windows)), dtype=np.float32)
    for _, g in meta.groupby(keys, sort=False, observed=True):
        t = g["time"].values
        srt = np.sort(t)
        pos = g.index.values
        upto = np.searchsorted(srt, t, side="right")
        for k, w in enumerate(windows):
            out[pos, k] = upto - np.searchsorted(srt, t - w, side="left")
    return out


def _sec(w):
    return f"{w // 60}m" if w >= 60 else f"{w}s"


def context_table(meta):
    """Все числовые признаки контекста для всей таблицы алертов (строка = rid)."""
    meta = meta.reset_index(drop=True)
    net = _window_counts(meta, ["scenario", "description"], BURST_WINDOWS)
    src = _window_counts(meta, ["scenario", "source"], ACT_WINDOWS)
    f = {}
    for k, w in enumerate(BURST_WINDOWS):
        f[f"same_event_{_sec(w)}"] = np.log1p(net[:, k])
    i60, i300, i3600 = BURST_WINDOWS.index(60), BURST_WINDOWS.index(300), BURST_WINDOWS.index(3600)
    f["ratio_1m_to_hour"] = net[:, i60] / (net[:, i3600] / 60.0 + 1.0)
    f["ratio_5m_to_hour"] = net[:, i300] / (net[:, i3600] / 12.0 + 1.0)
    for k, w in enumerate(ACT_WINDOWS):
        f[f"source_all_{_sec(w)}"] = np.log1p(src[:, k])
    return pd.DataFrame(f).astype(np.float32)


BURST_COLS = [f"same_event_{_sec(w)}" for w in BURST_WINDOWS] + ["ratio_1m_to_hour", "ratio_5m_to_hour"]
ACT_COLS = [f"source_all_{_sec(w)}" for w in ACT_WINDOWS]


# --- сборка матрицы ---------------------------------------------------------------------------
class Featurizer:
    """Учится только на обучающих данных (словарь текста), потом переводит любые алерты в матрицу."""

    def __init__(self, groups=MAIN_GROUPS):
        self.groups = tuple(groups)

    def fit(self, df):
        if "text" in self.groups:
            txt = text_of(df, "mask_ip" in self.groups)
            self.word = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True).fit(txt)
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000,
                                        sublinear_tf=True).fit(txt)
        return self

    def names(self):
        n = []
        if "text" in self.groups:
            n += [f"слово: {w}" for w in self.word.get_feature_names_out()]
            n += [f"кусок: «{c}»" for c in self.char.get_feature_names_out()]
        if "bursts" in self.groups:
            n += BURST_COLS
        if "activity" in self.groups:
            n += ACT_COLS
        if "source" in self.groups:
            n += ["source_wazuh"]
        if "identity" in self.groups:
            n += [f"scenario_{s}" for s in SCENARIOS]
        return n

    def transform(self, df, ctx):
        """df — алерты (с колонкой rid), ctx — таблица context_table для всех алертов."""
        blocks = []
        if "text" in self.groups:
            txt = text_of(df, "mask_ip" in self.groups)
            blocks += [self.word.transform(txt), self.char.transform(txt)]
        num = []
        rid = df["rid"].values
        if "bursts" in self.groups:
            num.append(ctx[BURST_COLS].values[rid])
        if "activity" in self.groups:
            num.append(ctx[ACT_COLS].values[rid])
        if "source" in self.groups:
            num.append((df["source"].values == "wazuh").astype(np.float32)[:, None])
        if "identity" in self.groups:
            num.append(np.stack([(df["scenario"].values == s) for s in SCENARIOS], 1).astype(np.float32))
        if num:
            blocks.append(csr_matrix(np.hstack(num).astype(np.float32)))
        return hstack(blocks).tocsr().astype(np.float32)
