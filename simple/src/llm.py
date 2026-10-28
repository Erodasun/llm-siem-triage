# llm.py — всё про языковые модели: промпт тандема, вызов модели, разбор ответа.
# Языковые модели не обучаются: им дают инструкцию (промпт) и один пример хорошего ответа.
import json, os, re, urllib.request

LLMS = {   # имя: (как вызывать, модель, файл с готовыми объяснениями в старой версии)
    "Qwen 3.5 4B": ("ollama", "qwen3.5:4b-q4_K_M", "tandem_main__ollama__qwen3.5-4b-q4_K_M.csv"),
    "GPT-6-sol": ("polza", "openai/gpt-6-sol", "tandem_main__polza__openai-gpt-6-sol.csv"),
    "gpt-oss-120b": ("polza", "openai/gpt-oss-120b", "tandem_main__polza__openai-gpt-oss-120b.csv"),
    "Claude": ("manual", "claude (вручную)", "tandem_main__claude__manual.csv"),
}

SYSTEM = (
    "Ты — старший аналитик SOC. Классификатор машинного обучения уже отобрал этот алерт SIEM как вероятную атаку "
    "и передал его тебе. Твоя задача — объяснить аналитику-человеку, что произошло, почему это может быть опасно "
    "и что сделать в первую очередь, и оценить срочность. Пиши по-русски, коротко и конкретно, опираясь на текст "
    "алерта и фрагмент журнала. Если в тексте нет явных признаков атаки, честно напиши об этом в why_suspicious и "
    "поставь attack_signs = \"нет\". Ответь одним JSON-объектом строго с ключами "
    '"priority", "attack_signs", "what_happened", "why_suspicious", "recommendation". '
    "priority — только одно из слов medium, high, critical; attack_signs — только \"да\" или \"нет\". "
    "Больше ничего не выводи."
)

EXAMPLE = (
    "Пример: описание='Reverse Shell'; детали='session opened with bash -i'; вероятность атаки по ML 0,99.\n"
    'Ответ: {"priority": "critical", "attack_signs": "да", '
    '"what_happened": "На хосте открыта обратная командная оболочка — злоумышленник получил интерактивный доступ", '
    '"why_suspicious": "bash -i с подключением наружу — прямой признак компрометации", '
    '"recommendation": "Разорвать соединение, изолировать хост, найти способ запуска оболочки по журналам"}\n\n'
)


def prompt(row, proba, same_event, with_example=True):
    """same_event — сколько таких же событий было в этой сети за 1, 5 и 15 минут."""
    log = row["full_log"] if isinstance(row["full_log"], str) and row["full_log"] else "(фрагмент журнала отсутствует)"
    ctx = ", ".join(f"{int(n)} за {m} мин" for n, m in zip(same_event, (1, 5, 15)))
    return ((EXAMPLE if with_example else "") + "Теперь объясни этот алерт.\n"
            f"Описание: {row['description']}\nДетали: {log[:400]}\nКонтекст: {ctx}\n"
            f"Вероятность атаки по ML: {proba:.2f}\n")


def ask(how, model, user, max_tokens=1500):
    if how == "ollama":
        url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        body = {"model": model, "prompt": user, "system": SYSTEM, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 500, "num_ctx": 4096}}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read().decode())["response"]
    key = os.environ.get("POLZA_API_KEY")
    if not key:
        raise SystemExit("Нет ключа: задайте переменную окружения POLZA_API_KEY")
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    req = urllib.request.Request("https://polza.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read().decode())["choices"][0]["message"]["content"]


def parse(raw):
    """Достаём поля из ответа; терпим лишний текст вокруг JSON и неточные имена ключей."""
    out = {"priority": "", "attack_signs": "", "what_happened": "", "why_suspicious": "", "recommendation": ""}
    m = re.search(r"\{.*\}", raw or "", re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            k = str(k).lower()
            for f in ("priority", "attack", "what", "why", "rec"):
                if k.startswith(f):
                    key = {"attack": "attack_signs", "what": "what_happened", "why": "why_suspicious",
                           "rec": "recommendation"}.get(f, f)
                    out[key] = str(v).strip()
    else:
        for f, key in (("priority", "priority"), ("attack", "attack_signs"), ("what", "what_happened"),
                       ("why", "why_suspicious"), ("rec", "recommendation")):
            mm = re.search(r'"(' + f + r'[a-z_]*)"\s*:\s*"((?:[^"\\]|\\.)*)', raw or "")
            out[key] = mm.group(2) if mm else ""
    p = out["priority"].lower()
    out["priority"] = p if p in ("medium", "high", "critical") else "medium"
    out["attack_signs"] = "да" if out["attack_signs"].lower() in ("да", "yes") else ("нет" if out["attack_signs"] else "")
    return out
