"""Chargement des jeux de données bruts.

- Pump Sensor Data (Kaggle, ``sensor.csv``) : 52 capteurs échantillonnés à la minute,
  colonne ``machine_status`` ∈ {NORMAL, RECOVERING, BROKEN}.
- NASA C-MAPSS (``train_FD00x.txt``, ``test_FD00x.txt``, ``RUL_FD00x.txt``) :
  trajectoires moteur jusqu'à la panne, 3 réglages opérationnels et 21 capteurs.

Aucune transformation métier ici (pas d'imputation, pas de filtrage) : les loaders
renvoient la donnée brute typée et validée.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CMAPSS_DIR, PUMP_CSV, STATUSES

SENSOR_PREFIX = "sensor_"
STATUS_COL = "machine_status"

CMAPSS_OP_COLS = [f"op_{i}" for i in range(1, 4)]
CMAPSS_SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit", "cycle", *CMAPSS_OP_COLS, *CMAPSS_SENSOR_COLS]
CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


# --------------------------------------------------------------------------- Pump

def load_pump(path: str | Path = PUMP_CSV) -> pd.DataFrame:
    """Charge ``sensor.csv`` avec le timestamp en index trié.

    Les capteurs sont convertis en ``float32`` et ``machine_status`` en catégorie
    ordonnée. Lève une erreur si l'index contient des doublons ou si un état
    inconnu apparaît.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Télécharger 'pump_sensor_data' sur Kaggle "
            "et placer sensor.csv dans data/raw/."
        )

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.has_duplicates:
        n = int(df.index.duplicated().sum())
        raise ValueError(f"{n} timestamps dupliqués dans {path.name}")

    unknown = set(df[STATUS_COL].dropna().unique()) - set(STATUSES)
    if unknown:
        raise ValueError(f"États machine inconnus : {sorted(unknown)}")
    df[STATUS_COL] = pd.Categorical(df[STATUS_COL], categories=list(STATUSES))

    sensors = pump_sensor_columns(df)
    df[sensors] = df[sensors].astype("float32")
    return df


def pump_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes capteurs du dataset Pump, dans l'ordre du fichier."""
    return [c for c in df.columns if c.startswith(SENSOR_PREFIX)]


# ------------------------------------------------------------------------ C-MAPSS

def _read_cmapss_table(path: Path) -> pd.DataFrame:
    # Les fichiers NASA ont des espaces en fin de ligne → colonnes vides à supprimer.
    raw = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    if raw.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"{path.name} : {raw.shape[1]} colonnes lues, {len(CMAPSS_COLUMNS)} attendues"
        )
    raw.columns = CMAPSS_COLUMNS
    return raw.astype({"unit": "int32", "cycle": "int32"})


def load_cmapss(
    subset: str = "FD002",
    split: str = "train",
    data_dir: str | Path = CMAPSS_DIR,
) -> pd.DataFrame:
    """Charge un sous-ensemble C-MAPSS et ajoute la colonne ``rul`` (cycles restants).

    - ``split="train"`` : trajectoires complètes, ``rul = cycle_max(unit) - cycle``.
    - ``split="test"``  : trajectoires tronquées, la RUL finale vient de
      ``RUL_<subset>.txt`` → ``rul = rul_finale(unit) + cycle_max(unit) - cycle``.
    """
    if subset not in CMAPSS_SUBSETS:
        raise ValueError(f"subset doit être dans {CMAPSS_SUBSETS}")
    if split not in ("train", "test"):
        raise ValueError("split doit être 'train' ou 'test'")

    data_dir = Path(data_dir)
    df = _read_cmapss_table(data_dir / f"{split}_{subset}.txt")
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    last_cycle = df.groupby("unit")["cycle"].transform("max")
    if split == "train":
        df["rul"] = last_cycle - df["cycle"]
    else:
        final_rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None).iloc[:, 0]
        units = sorted(df["unit"].unique())
        if len(final_rul) != len(units):
            raise ValueError(
                f"RUL_{subset}.txt : {len(final_rul)} valeurs pour {len(units)} unités"
            )
        rul_by_unit = pd.Series(final_rul.to_numpy(), index=units)
        df["rul"] = df["unit"].map(rul_by_unit) + last_cycle - df["cycle"]

    df["rul"] = df["rul"].astype("int32")
    return df
