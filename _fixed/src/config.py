import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "ait_ads"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

LABELS_CSV = DATA_RAW / "labels.csv"
FULL = DATA_PROCESSED / "02_labeled_alerts.csv"
TRAIN = RESULTS / "stage_1" / "train_alerts.csv"
TEST = RESULTS / "stage_1" / "test_alerts.csv"
SAMPLE = RESULTS / "stage_2" / "experiment_sample.csv"
SAMPLE_CTX = RESULTS / "stage_2" / "experiment_sample_ctx.csv"

SEED = 42
SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]
TEST_ATTACKS = ["webshell", "cracking", "reverse_shell", "privilege_escalation", "service_stop", "dnsteal"]

LEVELS = ["low", "medium", "high", "critical"]
LEVEL = {name: i for i, name in enumerate(LEVELS)}
CRIT = {
    "benign": "low", "network_scans": "low", "service_scans": "low", "dirb": "low",
    "wpscan": "medium", "service_stop": "medium",
    "cracking": "high", "dnsteal": "high",
    "webshell": "critical", "reverse_shell": "critical", "privilege_escalation": "critical",
}
RU_LEVELS = {
    "низкий": "low", "низкая": "low", "средний": "medium", "средняя": "medium",
    "высокий": "high", "высокая": "high", "критический": "critical", "критическая": "critical",
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
PROVIDERS = {
    "polza": {"api": "https://polza.ai/api/v1/chat/completions", "key_env": "POLZA_API_KEY"},
    "tokify": {"api": "https://tokify.sale/v1/chat/completions", "key_env": "TOKIFY_API_KEY"},
}


def stage_dir(n):
    path = RESULTS / f"stage_{n}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def utf8_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
