# run_all.ps1 — запуск простой версии целиком, по порядку. Из корня проекта:
#   powershell -ExecutionPolicy Bypass -File simple\run_all.ps1
# Платные запросы к облачным LLM здесь НЕ выполняются: недостающие объяснения запрашиваются отдельно
# командой  python simple\4_tandem.py --ask "gpt-oss-120b"  (см. вывод шага 4).
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
& $py simple\1_data.py
& $py simple\2_samples.py
& $py simple\3_ml.py
& $py simple\4_tandem.py
& $py simple\5_eval.py
& $py simple\6_dashboard.py
Write-Host "Готово. Итоговые таблицы: simple\results\итог.xlsx, дашборд: simple\results\dashboard.html"
