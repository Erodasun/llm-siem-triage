$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$old = Join-Path $root "_old"
New-Item -ItemType Directory -Force -Path $old | Out-Null

$items = @(
    "results\stage_2\console_10_llm.txt",
    "results\stage_2\experiment_sample_anomaly.csv",
    "results\stage_2\experiment_sample_freq.csv",
    "results\stage_2\llm_predictions.csv",
    "results\stage_2\ml_rf_predictions.csv",
    "results\stage_2\04_rf_vs_llm.png",
    "results\stage_4",
    "results\stage_5\hybrid_llm_explanations.csv",
    "results\stage_6",
    "results\stage_7",
    "results\stage_8",
    "reports\REPORT_SUMMARY.md",
    "reports\Отчет_по_практике_LLM_триаж_полный.docx",
    "reports\figures\dash_ensemble.png",
    "reports\figures\dash_freq.png",
    "reports\figures\fig17_dashboard_gemini.png",
    "reports\vkr_draft.md",
    "reports\vkr_draft.docx",
    "data\raw\ait_ads.zip",
    "new"
)

foreach ($rel in $items) {
    $src = Join-Path $root $rel
    if (Test-Path $src) {
        $dst = Join-Path $old $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Move-Item -Path $src -Destination $dst -Force
        Write-Host "moved: $rel"
    }
}
Write-Host "Готово. Всё перенесено в $old — после проверки папку можно удалить."
