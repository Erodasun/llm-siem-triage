# LLM-триаж алертов SIEM (AIT-ADS + Ollama/облачные LLM)

Прототип приоритизации алертов SIEM: классический ML, большие языковые модели (LLM) и их тандемы
расставляют приоритет (`low → medium → high → critical`), LLM дополнительно даёт объяснение и рекомендацию
на русском. Результаты выводятся в веб-дашборд триажа.

Разбор найденных ошибок и план доработки — `docs/AUDIT.md`.

## 1. Окружение

| Компонент | Версия |
|---|---|
| ОС | Windows 11 (x64) |
| Python | 3.13 (venv `.venv`) |
| ML | scikit-learn, xgboost, lightgbm |
| LLM локально | Ollama, `http://127.0.0.1:11434`, `qwen3.5:4b-q4_K_M` и др. |
| LLM облако | Polza AI, Tokify (OpenAI-совместимый API) |
| Дашборд | Flask, `http://127.0.0.1:5001` |

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3.5:4b-q4_K_M
$env:POLZA_API_KEY = "..."      # ключи только через переменные окружения
$env:TOKIFY_API_KEY = "..."
```

Все скрипты запускаются из корня проекта через `.venv\Scripts\python.exe`. Пути берутся из `src/config.py`
(корень проекта определяется автоматически), общие функции — `src/common.py`, единый промпт — `src/prompts.py`.

## 2. Данные

AIT-ADS (`data/raw/ait_ads/`): по 8 сценариям пары логов Wazuh/Suricata и AMiner + `labels.csv`
с интервалами атак. Разметка временная: алерт внутри интервала атаки получает её метку, иначе `benign`.
Разбиение train/test хронологическое: граница сценария — первый алерт фазы `webshell`.

## 3. Порядок запуска

| Шаг | Скрипт | Что делает | Выход |
|---|---|---|---|
| 1 | `03_build_dataset.py` | Разметка и сборка датасета | `data/processed/02_labeled_alerts.csv` |
| 2 | `04_split_dataset.py` | Хронологический split | `results/stage_1/` |
| 3 | `08_prepare_experiment_sample.py` | Выборка 200 алертов в 4 группах: лёгкие атаки (тип редок в фоне), трудные атаки (тип есть и в фоне), парный фон с тем же текстом, случайный фон; с `time`, `alert_id`, `group` | `results/stage_2/experiment_sample.csv` |
| 4 | `13_stage3_ml_models.py` | 5 ML на TF-IDF, полный тест | `results/stage_3/` |
| 5 | `16_stage5_hybrid.py` | ML-фильтр: доля отсеянного потока и потерянных атак | `results/stage_5/` |
| 6 | `26_context_features.py` | Частотный контекст для выборки | `results/stage_2/experiment_sample_ctx.csv` |
| 7 | `14_stage4_llm_runner.py` | Прогон LLM (локально/облако, zero/few-shot, с контекстом) | `results/stage_4/llm__*.csv` |
| 8 | `43_build_tandem_study.py` | ML на богатых признаках, перебор «ML × LLM», тандемы | `results/stage_7/` |
| 9 | `42_build_dashboard_models.py` | Файлы ML-only и «ML решает → LLM объясняет» | `results/stage_7/` |
| 10 | `39_all_models_attack_metrics.py` | Единая таблица метрик + базовая линия «всё — атака» | `results/stage_7/all_models_attack_metrics.csv` |
| 11 | `17_stage6_flask_app.py --static-only` | Статичный дашборд | `results/stage_6/dashboard.html` |
| 11a | `46_dashboard_screenshots.py` (нужен playwright с Chromium) | Снимки дашборда для отчёта | `reports/figures/dash_*.png` |
| 12 | `24_build_report_figures.py`, `23_build_practice_report.py` | Рисунки и docx-отчёт | `reports/` |
| 13 | `44_second_sample_check.py` | Вторая независимая выборка и ML-модели на ней | `results/stage_8/` |
| 14 | `run_second_sample_llm.ps1` (запускает `14 --sample2`, затем `45_second_sample_llm_report.py`) | Локальные LLM и Тандем 1 на второй выборке | `results/stage_8/all_models_sample2.csv` |

Папки результатов (номер совпадает с номером в имени скрипта, где он есть):

| Папка | Содержимое |
|---|---|
| `results/stage_1/` | train/test, распределения |
| `results/stage_2/` | оценочная выборка (200 алертов) и её частотный контекст |
| `results/stage_3/` | ML на тексте, полный тест |
| `results/stage_4/` | прогоны LLM (`llm__*.csv`) |
| `results/stage_5/` | гибридный текстовый ML-фильтр |
| `results/stage_6/` | дашборд |
| `results/stage_7/` | ML на богатых признаках, тандемы, единая таблица метрик |
| `results/stage_8/` | вторая оценочная выборка: ML, LLM, Тандем 1 |

Примеры прогона LLM:

```powershell
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model qwen3.5:4b-q4_K_M --shots few --limit 5
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model qwen3.5:4b-q4_K_M --shots few --context freq
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend polza --model openai/gpt-6-sol
```

Прогон продолжается с места остановки (resume), строки с ошибкой перезапрашиваются.

## 4. Метрика

Атака = приоритет `medium`, `high` или `critical`. Ответ с ошибкой (`priority = error`) считается пропуском
и выводится отдельно. В каждой сводке есть тривиальная базовая линия «всё — атака»: на текущей выборке (87 атак из 200) она
даёт F1 = 0.606, и любая модель должна сравниваться с ней. Порог ML выбирается на временной валидации
из train, а не на оценочной выборке.

## 5. Структура

- `scripts/` — пайплайн (номер = этап);
- `src/` — конфигурация, общие функции, промпт;
- `results/stage_N/` — результаты этапов;
- `reports/` — отчёт, рисунки, пояснение для научного руководителя;
- `docs/AUDIT.md` — аудит и план доработки.

## 6. Репозиторий

В репозиторий входит только код (`scripts/`, `src/`, `README.md`, `requirements.txt`, `docs/`).
Каталоги `data/`, `results/`, `reports/`, `.venv/`, `_old/`, `logs/` и сгенерированные файлы
(`*.docx`, `*.pdf`, `*.xlsx`) исключены через `.gitignore` — они большие и воспроизводятся запуском пайплайна.

Чтобы получить данные: положить AIT-ADS в `data/raw/ait_ads/` (см. §2) и выполнить шаги из раздела 3.
Ключи задаются только через переменные окружения (`POLZA_API_KEY`, `TOKIFY_API_KEY`) и в репозиторий не попадают.
