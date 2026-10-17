import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from src.config import LEVEL, LEVELS, OLLAMA_URL, PROVIDERS, RU_LEVELS

POSITIVE = {"medium", "high", "critical"}
FIELDS = ("what_happened", "why_suspicious", "recommendation")


def alert_text(df, n=300):
    return df["description"].fillna("").astype(str) + " " + df["full_log"].fillna("").astype(str).str[:n]


def log_snippet(value, n):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "(фрагмент журнала отсутствует)"
    return str(value)[:n]


def normalize_priority(value):
    if not isinstance(value, str):
        return "error"
    p = value.strip().lower()
    if p in LEVEL:
        return p
    return RU_LEVELS.get(p, "error")


def _field_by_prefix(obj, prefix):
    for k, v in obj.items():
        if isinstance(k, str) and k.lower().startswith(prefix) and v:
            return str(v)
    return ""


def _regex_field(text, prefix):
    m = re.search(r'"(' + prefix + r'[a-z_]*)"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    return m.group(2) if m else ""


def parse_llm_json(text):
    out = {"priority": "error", "what_happened": "", "why_suspicious": "", "recommendation": "", "parse_ok": False}
    if not isinstance(text, str) or not text.strip():
        return out
    obj = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    if isinstance(obj, dict):
        out["priority"] = normalize_priority(obj.get("priority"))
        out["what_happened"] = _field_by_prefix(obj, "what")
        out["why_suspicious"] = _field_by_prefix(obj, "why")
        out["recommendation"] = _field_by_prefix(obj, "rec")
        out["parse_ok"] = out["priority"] != "error"
        return out
    pm = re.search(r'"priority"\s*:\s*"?([A-Za-zА-Яа-я]+)', text)
    if pm:
        out["priority"] = normalize_priority(pm.group(1))
    out["what_happened"] = _regex_field(text, "what")
    out["why_suspicious"] = _regex_field(text, "why")
    out["recommendation"] = _regex_field(text, "rec")
    return out


def ollama_generate(model, prompt, system=None, num_predict=400, num_ctx=4096, timeout=600):
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False,
               "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx}}
    if system:
        payload["system"] = system
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def chat_completion(provider, model, system, user, max_tokens=600, timeout=600):
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"], "")
    if not key:
        raise SystemExit(f"{cfg['key_env']} not set")
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(cfg["api"], data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def time_index(df, keys):
    return {k: np.sort(v.values) for k, v in df.groupby(list(keys), sort=False, dropna=False)["time"]}


def burst_counts(index, key_tuples, times, window):
    out = np.zeros(len(times), dtype=np.int64)
    for j, (k, t) in enumerate(zip(key_tuples, times)):
        arr = index.get(k)
        if arr is not None:
            out[j] = np.searchsorted(arr, t, side="right") - np.searchsorted(arr, t - window, side="left")
    return out


def is_positive(priority):
    return pd.Series(priority).astype(str).str.lower().isin(POSITIVE).values


def level_of(priority):
    return pd.Series(priority).astype(str).str.lower().map(LEVEL).fillna(0).astype(int).values


def attack_metrics(y_attack, pred):
    y = np.asarray(y_attack, dtype=bool)
    p = np.asarray(pred, dtype=bool)
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"found": tp, "fp": fp, "recall": round(rec, 3), "prec": round(prec, 3),
            "F1": round(f1, 3), "acc": round(float((p == y).mean()), 3)}


def trivial_baseline(y_attack):
    return attack_metrics(y_attack, np.ones(len(y_attack), dtype=bool))


def levels_from_score(score, flagged, q_crit=0.85, q_high=0.60):
    out = np.array(["low"] * len(score), dtype=object)
    idx = np.where(flagged)[0]
    if len(idx) == 0:
        return out
    s = np.asarray(score)[idx]
    c, h = np.quantile(s, [q_crit, q_high])
    out[idx] = np.select([s >= c, s >= h], ["critical", "high"], "medium")
    return out


__all__ = ["LEVELS", "POSITIVE", "FIELDS", "alert_text", "log_snippet", "normalize_priority", "parse_llm_json",
           "ollama_generate", "chat_completion", "time_index", "burst_counts", "is_positive", "level_of",
           "attack_metrics", "trivial_baseline", "levels_from_score"]
