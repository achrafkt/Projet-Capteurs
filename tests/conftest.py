import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def pump_frame() -> pd.DataFrame:
    """Petit jeu Pump synthétique : 2 pannes, 1 capteur mort, 1 capteur figé."""
    idx = pd.date_range("2018-04-01", periods=10 * 24 * 60, freq="min")
    rng = np.random.default_rng(0)
    status = np.array(["NORMAL"] * len(idx), dtype=object)
    status[3 * 1440] = "BROKEN"
    status[3 * 1440 + 1: 3 * 1440 + 600] = "RECOVERING"
    status[7 * 1440: 7 * 1440 + 2] = "BROKEN"
    status[7 * 1440 + 2: 7 * 1440 + 300] = "RECOVERING"

    frozen = rng.normal(size=len(idx))
    frozen[1000: 1000 + 2000] = 3.0
    return pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_00": rng.normal(size=len(idx)),
            "sensor_01": np.nan,
            "sensor_02": frozen,
            "machine_status": status,
        }
    )


@pytest.fixture
def pump_csv(tmp_path, pump_frame) -> Path:
    path = tmp_path / "sensor.csv"
    pump_frame.to_csv(path)  # l'index devient la colonne « Unnamed: 0 » comme sur Kaggle
    return path
