# Запуск локальных LLM на второй выборке (бесплатно, Ollama должен быть запущен). Порядок: быстрые -> медленные.
# Каждый запуск возобновляемый: при обрыве просто запустить скрипт снова.
$ErrorActionPreference = "Continue"
Set-Location "D:\Default Project"
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model gemma2:2b --sample2
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model llama3.2:3b --sample2
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model qwen3.5:4b-q4_K_M --context freq --sample2
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model qwen3.5:4b-q4_K_M --sample2
.venv\Scripts\python.exe scripts\14_stage4_llm_runner.py --backend ollama --model llama3.1:latest --sample2
.venv\Scripts\python.exe scripts\45_second_sample_llm_report.py
