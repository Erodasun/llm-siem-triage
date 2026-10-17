from src.common import log_snippet

SYSTEM = (
    "Ты — старший аналитик SOC, выполняешь первичный триаж алертов SIEM. "
    "Ответь одним JSON-объектом на русском языке строго с ключами "
    '"priority", "what_happened", "why_suspicious", "recommendation". '
    "Значение priority — только одно из английских слов low, medium, high, critical. "
    "Больше ничего не выводи."
)

FEW_SHOT = (
    "Пример 1: описание='Reverse Shell'; детали='session opened with bash -i'.\n"
    'Ответ: {"priority": "critical", "what_happened": "Обнаружена обратная командная оболочка — активная C2-сессия", '
    '"why_suspicious": "Прямой признак компрометации: злоумышленник получил интерактивный доступ к хосту", '
    '"recommendation": "Заблокировать соединение с C2-адресом, изолировать хост, проверить журналы оболочки"}\n\n'
    "Пример 2: описание='bruteforce ssh: 50 login failures'; детали='many Invalid user from 10.0.0.5'.\n"
    'Ответ: {"priority": "high", "what_happened": "Множественные неудачные попытки входа по SSH", '
    '"why_suspicious": "Перебор паролей — типичный шаг перед компрометацией", '
    '"recommendation": "Ограничить доступ по IP, включить защиту от перебора, проверить успешные входы за тот же период"}\n\n'
    "Пример 3: описание='Wazuh agent started'; детали='wazuh-agentd: INFO: Agent started'.\n"
    'Ответ: {"priority": "low", "what_happened": "Штатный запуск агента мониторинга", '
    '"why_suspicious": "Служебное событие без признаков атаки", '
    '"recommendation": "Действий не требуется"}\n\n'
)

CONTEXT_HINT = {
    "freq": ("В поле «Контекст» указано, сколько таких же событий пришло за последние минуты. "
             "Всплеск одинаковых событий может указывать на автоматизированную атаку, но фон тоже бывает шумным — "
             "оценивай совокупно.\n"),
    "anomaly": ("В поле «Контекст» указано, сколько таких же событий пришло за последние минуты и во сколько раз это "
                "больше обычного фона для этого события. Значение около 1 — норма; оценивай отклонение, "
                "а не абсолютное число.\n"),
}

WINDOWS_MIN = (60, 300, 900)


def context_line(row, mode):
    if mode == "freq":
        return ", ".join(f"{int(row[f'burst_{w}'])} за {w // 60} мин" for w in WINDOWS_MIN)
    if mode == "anomaly":
        return "; ".join(f"{int(row[f'burst_{w}'])} за {w // 60} мин (обычно {row[f'bg_{w}']:.0f}, "
                         f"в {row[f'ratio_{w}']:.1f} раза)" for w in WINDOWS_MIN)
    return ""


def build_prompt(row, shots="zero", context="none", log_chars=400):
    parts = []
    if context != "none":
        parts.append(CONTEXT_HINT[context])
    if shots == "few":
        parts.append(FEW_SHOT)
        parts.append("Теперь проанализируй это событие.\n")
    parts.append(f"Описание: {row['description']}\nДетали: {log_snippet(row['full_log'], log_chars)}\n")
    if context != "none":
        parts.append(f"Контекст: {context_line(row, context)}\n")
    return "".join(parts)
