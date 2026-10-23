"""Profil de qualité des capteurs et diagnostic des capteurs morts ou suspects."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import (
    DEAD_MODE_RATIO,
    DEAD_NAN_RATIO,
    STATIONARITY_ALPHA,
    STATIONARITY_RESAMPLE,
    SUSPECT_FROZEN,
    SUSPECT_NAN_RATIO,
)


def time_index_report(index: pd.DatetimeIndex) -> pd.Series:
    """Régularité de l'échantillonnage : pas médian, trous, doublons."""
    steps = index.to_series().diff().dropna()
    step = steps.median()
    gaps = steps[steps > step]
    return pd.Series(
        {
            "n_points": len(index),
            "start": index.min(),
            "end": index.max(),
            "median_step": step,
            "is_monotonic": index.is_monotonic_increasing,
            "n_duplicates": int(index.duplicated().sum()),
            "n_gaps": len(gaps),
            "largest_gap": gaps.max() if len(gaps) else pd.Timedelta(0),
            "missing_points": int(((gaps / step) - 1).sum()) if len(gaps) else 0,
        }
    )


def longest_constant_run(s: pd.Series) -> tuple[int, pd.Timestamp | None]:
    """Plus longue séquence de valeurs identiques consécutives (NaN exclus).

    Renvoie (nombre de points, timestamp de début). Une valeur isolée compte pour 1.
    """
    v = s.to_numpy()
    valid = ~pd.isna(v)
    if not valid.any():
        return 0, None
    same = np.zeros(len(v), dtype=bool)
    same[1:] = (v[1:] == v[:-1]) & valid[1:] & valid[:-1]
    if not same.any():
        return 1, s.index[np.flatnonzero(valid)[0]]
    edges = np.flatnonzero(np.diff(np.concatenate(([0], same.astype(np.int8), [0]))))
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    # same[i] = True signifie v[i] == v[i-1] : la séquence commence un point avant.
    return int(ends[k] - starts[k] + 1), s.index[starts[k] - 1]


def stationarity_tests(
    s: pd.Series,
    resample: str | None = STATIONARITY_RESAMPLE,
    alpha: float = STATIONARITY_ALPHA,
) -> dict:
    """ADF (H0 : racine unitaire) et KPSS (H0 : stationnarité en niveau).

    La série est agrégée (moyenne sur ``resample``) avant les tests : à la minute,
    l'ADF sur ~220 000 points est lent et quasi toujours significatif.
    """
    x = s.resample(resample).mean().dropna() if resample else s.dropna()
    out = {"adf_pvalue": np.nan, "kpss_pvalue": np.nan, "stationarity": "non testable"}
    if len(x) < 30 or np.isclose(x.std(), 0):
        return out

    out["adf_pvalue"] = adfuller(x.to_numpy(), autolag="AIC")[1]
    with warnings.catch_warnings():
        # KPSS borne ses p-values à [0.01, 0.1] et prévient par un warning.
        warnings.simplefilter("ignore", InterpolationWarning)
        out["kpss_pvalue"] = kpss(x.to_numpy(), regression="c", nlags="auto")[1]

    adf_stationary = out["adf_pvalue"] < alpha
    kpss_stationary = out["kpss_pvalue"] >= alpha
    if adf_stationary and kpss_stationary:
        out["stationarity"] = "stationnaire"
    elif not adf_stationary and not kpss_stationary:
        out["stationarity"] = "non stationnaire"
    elif adf_stationary:
        out["stationarity"] = "contradictoire (ruptures / régimes probables)"
    else:
        out["stationarity"] = "indéterminé"
    return out


def profile_sensors(
    df: pd.DataFrame,
    sensors: list[str],
    resample: str | None = STATIONARITY_RESAMPLE,
) -> pd.DataFrame:
    """Profil de qualité, une ligne par capteur.

    - ``nan_pct``         : part de valeurs manquantes
    - ``mode_pct``        : part des mesures égales à la valeur la plus fréquente
    - ``zero_step_pct``   : part des pas consécutifs sans variation (capteur figé / quantifié)
    - ``longest_frozen``  : plus longue séquence de valeurs identiques (durée)
    """
    step = df.index.to_series().diff().median()
    rows = []
    for name in sensors:
        s = df[name]
        valid = s.dropna()
        diffs = s.diff()
        n_steps = int(diffs.notna().sum())
        run_len, run_start = longest_constant_run(s)
        rows.append(
            {
                "sensor": name,
                "nan_pct": 100 * s.isna().mean(),
                "mean": valid.mean(),
                "std": valid.std(),
                "min": valid.min(),
                "max": valid.max(),
                "iqr": valid.quantile(0.75) - valid.quantile(0.25),
                "n_unique": valid.nunique(),
                "mode_pct": 100 * valid.value_counts(normalize=True).iloc[0] if len(valid) else np.nan,
                "zero_step_pct": 100 * (diffs == 0).sum() / n_steps if n_steps else np.nan,
                "longest_frozen": run_len * step,
                "longest_frozen_start": run_start,
                **stationarity_tests(s, resample=resample),
            }
        )
    return pd.DataFrame(rows).set_index("sensor")


def classify_sensors(
    profile: pd.DataFrame,
    dead_nan_ratio: float = DEAD_NAN_RATIO,
    dead_mode_ratio: float = DEAD_MODE_RATIO,
    suspect_nan_ratio: float = SUSPECT_NAN_RATIO,
    suspect_frozen: str | pd.Timedelta = SUSPECT_FROZEN,
) -> pd.DataFrame:
    """Statut ``mort`` / ``suspect`` / ``ok`` et justification pour chaque capteur.

    Mort (exclu de la suite) : >= ``dead_nan_ratio`` de NaN, variance nulle,
    ou une seule valeur couvrant >= ``dead_mode_ratio`` des mesures.
    Suspect (gardé, traité au nettoyage) : >= ``suspect_nan_ratio`` de NaN,
    ou figé pendant >= ``suspect_frozen``.
    """
    suspect_frozen = pd.Timedelta(suspect_frozen)
    status, reasons = [], []
    for _, p in profile.iterrows():
        dead, suspect = [], []
        if p.nan_pct >= 100 * dead_nan_ratio:
            dead.append(f"{p.nan_pct:.1f} % de NaN (seuil {100 * dead_nan_ratio:.0f} %)")
        if p.nan_pct < 100 and (pd.isna(p["std"]) or np.isclose(p["std"], 0)):
            dead.append("variance nulle")
        if p.mode_pct >= 100 * dead_mode_ratio:
            dead.append(f"une valeur couvre {p.mode_pct:.1f} % des mesures")
        if not dead:
            if p.nan_pct >= 100 * suspect_nan_ratio:
                suspect.append(f"{p.nan_pct:.1f} % de NaN")
            if p.longest_frozen >= suspect_frozen:
                suspect.append(
                    f"figé {p.longest_frozen} à partir du {p.longest_frozen_start:%Y-%m-%d %H:%M}"
                )
        status.append("mort" if dead else "suspect" if suspect else "ok")
        reasons.append(" ; ".join(dead or suspect))
    out = profile.copy()
    out["status"] = status
    out["justification"] = reasons
    return out
