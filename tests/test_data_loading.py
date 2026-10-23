import numpy as np
import pandas as pd
import pytest

from src.data_loading import CMAPSS_COLUMNS, load_cmapss, load_pump, pump_sensor_columns


def test_load_pump_index_and_types(pump_csv):
    df = load_pump(pump_csv)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert "Unnamed: 0" not in df.columns
    assert pump_sensor_columns(df) == ["sensor_00", "sensor_01", "sensor_02"]
    assert df["sensor_00"].dtype == np.float32
    assert list(df["machine_status"].cat.categories) == ["NORMAL", "RECOVERING", "BROKEN"]


def test_load_pump_rejects_unknown_status(tmp_path, pump_frame):
    pump_frame.loc[5, "machine_status"] = "BROKE"
    path = tmp_path / "sensor.csv"
    pump_frame.to_csv(path)
    with pytest.raises(ValueError, match="inconnus"):
        load_pump(path)


def test_load_pump_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pump(tmp_path / "absent.csv")


def _write_cmapss(directory, split, units_cycles):
    lines = []
    for unit, n in units_cycles:
        for cycle in range(1, n + 1):
            values = [unit, cycle] + [0.5] * (len(CMAPSS_COLUMNS) - 2)
            lines.append(" ".join(str(v) for v in values) + "  ")  # espaces finaux NASA
    (directory / f"{split}_FD002.txt").write_text("\n".join(lines) + "\n")


def test_load_cmapss_train_rul(tmp_path):
    _write_cmapss(tmp_path, "train", [(1, 5), (2, 3)])
    df = load_cmapss("FD002", "train", tmp_path)
    assert list(df.columns) == CMAPSS_COLUMNS + ["rul"]
    assert df.loc[df.unit == 1, "rul"].tolist() == [4, 3, 2, 1, 0]
    assert df.loc[df.unit == 2, "rul"].tolist() == [2, 1, 0]


def test_load_cmapss_test_rul(tmp_path):
    _write_cmapss(tmp_path, "test", [(1, 3), (2, 2)])
    (tmp_path / "RUL_FD002.txt").write_text("10 \n20 \n")
    df = load_cmapss("FD002", "test", tmp_path)
    assert df.loc[df.unit == 1, "rul"].tolist() == [12, 11, 10]
    assert df.loc[df.unit == 2, "rul"].tolist() == [21, 20]
