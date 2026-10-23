"""Phase 2 : nettoyage des données Pump.

Règles suivies (voir reports/protocole_evaluation.md) :
- aucune ligne n'est supprimée : l'index temporel reste intact ;
- les valeurs extrêmes sont conservées (ce sont des anomalies potentielles) ;
- aucune statistique apprise sur les données (moyenne, médiane, écart-type)
  n'est calculée ici : l'imputation apprise se fait dans chaque pli (phase 4) ;
- le remplissage est causal : on ne reporte que la dernière valeur connue,
  jamais une valeur future, et jamais à travers un changement de statut.

Ce que fait le nettoyage, dans cet ordre :
1. retirer les capteurs morts ;
2. masquer (NaN) les séquences figées plus longues que `frozen_min` ;
3. remplir les trous courts (au plus `ffill_limit` minutes) ;
4. lister les capteurs utilisables par les modèles multivariés.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STATUS_COL = "machine_status"
FROZEN_MIN = pd.Timedelta("24h")  # une séquence identique plus longue est « figée »
FFILL_LIMIT = 5  # nombre maximal de minutes remplies par la dernière valeur connue
FEATURE_MAX_NAN_FRAC = 0.20  # au-delà, le capteur n'est pas utilisé comme variable


@dataclass
class CleaningResult:
    data: pd.DataFrame  # statut + capteurs conservés, nettoyés (NaN restants possibles)
    frozen: pd.DataFrame  # True là où une valeur figée a été masquée
    dropped: dict[str, str]  # capteur retiré -> raison
    features: list[str]  # capteurs utilisables par les modèles multivariés
    report: pd.DataFrame  # une ligne par capteur conservé


def mask_frozen(
    df: pd.DataFrame,
    sensors: list[str],
    min_duration: pd.Timedelta = FROZEN_MIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remplace par NaN les séquences de valeurs identiques d'au moins `min_duration`.

    Retourne (données masquées, indicateurs booléens des points masqués).
    Les NaN existants ne sont jamais considérés comme « figés ».
    """
    sensors = list(sensors)
    step = df.index.to_series().diff().median()
    min_points = int(np.ceil(min_duration / step))

    flags = pd.DataFrame(False, index=df.index, columns=sensors)
    for col in sensors:
        s = df[col]
        run_id = s.ne(s.shift()).cumsum()  # un nouvel identifiant à chaque changement de valeur
        run_len = s.groupby(run_id).transform("size")  # longueur de la séquence de chaque point
        flags[col] = (run_len >= min_points) & s.notna()

    return df[sensors].mask(flags), flags


def fill_short_gaps(
    df: pd.DataFrame,
    sensors: list[str],
    status: pd.Series,
    limit: int = FFILL_LIMIT,
) -> pd.DataFrame:
    """Remplit les trous courts avec la dernière valeur connue (remplissage causal).

    - au plus `limit` points consécutifs sont remplis ;
    - rien n'est reporté à travers un changement de statut (NORMAL, BROKEN, RECOVERING) ;
    - un trou en début de série reste NaN (on n'utilise jamais le futur).
    """
    sensors = list(sensors)
    segment = status.ne(status.shift()).cumsum()
    return df[sensors].groupby(segment).ffill(limit=limit)


def clean_pump(
    df: pd.DataFrame,
    sensors: list[str],
    dead_sensors: list[str],
    status_col: str = STATUS_COL,
    frozen_min: pd.Timedelta = FROZEN_MIN,
    ffill_limit: int = FFILL_LIMIT,
    feature_max_nan_frac: float = FEATURE_MAX_NAN_FRAC,
) -> CleaningResult:
    """Applique le nettoyage complet et retourne les données et un rapport."""
    dropped = {s: "mort (capteur inutilisable)" for s in dead_sensors}
    kept = [s for s in sensors if s not in dropped]

    raw = df[kept]
    masked, frozen = mask_frozen(df, kept, frozen_min)
    filled = fill_short_gaps(masked, kept, df[status_col], ffill_limit)
    filled = filled.mask(frozen)  # le remplissage ne doit pas recréer de valeurs figées

    nan_final = filled.isna().mean()
    features = [s for s in kept if nan_final[s] <= feature_max_nan_frac]

    report = pd.DataFrame(
        {
            "nan_brut_pct": raw.isna().mean() * 100,
            "fige_masque_pct": frozen.mean() * 100,
            "rempli_pct": (raw.isna() & filled.notna()).mean() * 100,
            "nan_final_pct": nan_final * 100,
        }
    ).round(3)
    report["statut"] = np.where(
        report.index.isin(features),
        "ok",
        f"exclu des variables (NaN > {feature_max_nan_frac:.0%})",
    )

    data = pd.concat([df[[status_col]], filled], axis=1)
    return CleaningResult(data, frozen, dropped, features, report)


def load_clean_pump(path) -> pd.DataFrame:
    """Recharge le fichier produit par le notebook 02 (data/processed/pump_clean.csv.gz)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)
