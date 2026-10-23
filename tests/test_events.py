import pandas as pd

from src.data_loading import load_pump
from src.events import extract_failure_events, pre_failure_mask, time_zones


def test_extract_failure_events(pump_csv):
    df = load_pump(pump_csv)
    events = extract_failure_events(df["machine_status"])
    assert len(events) == 2
    t0 = pd.Timestamp("2018-04-01")
    assert events[0].failure_time == t0 + pd.Timedelta(days=3)
    assert events[0].recovery_start == t0 + pd.Timedelta(days=3, minutes=1)
    assert events[0].recovery_end == t0 + pd.Timedelta(days=3, minutes=599)
    assert events[1].failure_time == t0 + pd.Timedelta(days=7)


def test_pre_failure_mask_window(pump_csv):
    df = load_pump(pump_csv)
    events = extract_failure_events(df["machine_status"])
    mask = pre_failure_mask(df.index, events, "48h")
    assert mask.sum() == 2 * 48 * 60
    assert not mask[events[0].failure_time]
    assert mask[events[0].failure_time - pd.Timedelta("1min")]


def test_pre_failure_mask_clipped_after_recovery(pump_csv):
    df = load_pump(pump_csv)
    events = extract_failure_events(df["machine_status"])
    # Fenêtre de 5 jours : la 2e panne remonterait dans la remise en service de la 1re.
    mask = pre_failure_mask(df.index, events, "5D")
    assert not mask[events[0].recovery_end]
    assert mask[events[0].recovery_end + pd.Timedelta("1min")]


def test_time_zones(pump_csv):
    df = load_pump(pump_csv)
    events = extract_failure_events(df["machine_status"])
    zones = time_zones(df["machine_status"], events, "48h")
    assert zones[events[0].failure_time] == "excluded"
    assert zones[events[0].recovery_end] == "excluded"
    assert zones[events[0].failure_time - pd.Timedelta("1h")] == "pre_failure"
    assert zones.iloc[-1] == "censored"
    assert zones.iloc[0] == "normal"
