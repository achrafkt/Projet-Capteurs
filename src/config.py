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

# Nettoyage (notebook 02)
FROZEN_MIN = SUSPECT_FROZEN  # une séquence identique au moins aussi longue est masquée
FFILL_LIMIT = 5              # nombre maximal de minutes remplies par la dernière valeur connue
FEATURE_MAX_NAN_FRAC = 0.20  # au-delà, le capteur n'est pas utilisé comme variable

# Segmentation des régimes (notebook 03)
REGIME_RESAMPLE = "10min"     # pas d'agrégation (médiane) avant segmentation
REGIME_K_RANGE = (2, 6)       # nombres de régimes testés (silhouette)
REGIME_PCA_VARIANCE = 0.90    # part de variance conservée par la PCA
REGIME_CLIP = 10.0            # borne des z-scores robustes avant PCA
REGIME_ENVELOPE_Q = 0.995     # quantile de distance au centre : au-delà, « hors régime »
REGIME_MIN_DURATION = "1h"    # un changement de régime n'est confirmé qu'après cette durée
REGIME_TRANSITION = "30min"   # zone masquée après chaque changement de régime confirmé

RANDOM_STATE = 42
