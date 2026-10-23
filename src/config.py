"""Chemins et paramètres partagés par tout le projet.

Les valeurs par défaut du protocole d'évaluation (reports/protocole_evaluation.md)
sont définies ici et nulle part ailleurs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

PUMP_CSV = DATA_RAW / "sensor.csv"
CMAPSS_DIR = DATA_RAW / "CMAPSS"

# États machine du dataset Pump
STATUS_NORMAL = "NORMAL"
STATUS_BROKEN = "BROKEN"
STATUS_RECOVERING = "RECOVERING"
STATUSES = (STATUS_NORMAL, STATUS_RECOVERING, STATUS_BROKEN)

# Protocole d'évaluation — Pump
PRE_FAILURE_WINDOW = "48h"
LOFO_PURGE = "24h"
ALERT_MERGE_GAP = "30min"
MAX_FALSE_ALARMS_PER_DAY = 1.0

# Protocole d'évaluation — C-MAPSS
CMAPSS_PRE_FAILURE_CYCLES = 30
CMAPSS_HEALTHY_RUL = 125

# Critères de qualité capteur (notebook 01)
DEAD_NAN_RATIO = 0.50       # capteur mort si >= 50 % de NaN
DEAD_MODE_RATIO = 0.95      # capteur mort si une seule valeur couvre >= 95 % des mesures
SUSPECT_NAN_RATIO = 0.20    # capteur suspect si >= 20 % de NaN
SUSPECT_FROZEN = "24h"      # capteur suspect si figé pendant >= 24 h d'affilée
STATIONARITY_RESAMPLE = "1h"
STATIONARITY_ALPHA = 0.05

RANDOM_STATE = 42
