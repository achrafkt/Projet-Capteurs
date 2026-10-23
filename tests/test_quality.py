import numpy as np
import pandas as pd

from src.data_loading import load_pump, pump_sensor_columns
from src.quality import classify_sensors, longest_constant_run, profile_sensors, time_index_report


def test_longest_constant_run():
    idx = pd.date_range("2020-01-01", periods=8, freq="min")
    s = pd.Series([1, 1, 2, 2, 2, 2, np.nan, np.nan], index=idx)
    assert longest_constant_run(s) == (4, idx[2])


def test_longest_constant_run_edge_cases():
    idx = pd.date_range("2020-01-01", periods=3, freq="min")
    assert longest_constant_run(pd.Series([np.nan] * 3, index=idx)) == (0, None)
    assert longest_constant_run(pd.Series([1.0, 2.0, 3.0], index=idx)) == (1, idx[0])


def test_time_index_report_detects_gap():
    idx = pd.date_range("2020-01-01", periods=10, freq="min").delete([4, 5])
    report = time_index_report(idx)
    assert report["n_gaps"] == 1
    assert report["missing_points"] == 2
    assert report["largest_gap"] == pd.Timedelta("3min")


def test_profile_and_classification(pump_csv):
    df = load_pump(pump_csv)
    profile = profile_sensors(df, pump_sensor_columns(df))
    classified = classify_sensors(profile)

    assert classified.loc["sensor_00", "status"] == "ok"
    assert classified.loc["sensor_00", "stationarity"] == "stationnaire"
    assert classified.loc["sensor_01", "status"] == "mort"
    assert "100.0 % de NaN" in classified.loc["sensor_01", "justification"]
    assert classified.loc["sensor_02", "status"] == "suspect"
    assert classified.loc["sensor_02", "longest_frozen"] == pd.Timedelta(minutes=2000)
