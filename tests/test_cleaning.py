"""Tests du nettoyage (données synthétiques, aucune donnée réelle requise)."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_pump, fill_short_gaps, mask_frozen


def make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2018-04-01", periods=n, freq="min")


def test_mask_frozen_masque_seulement_les_longues_sequences():
    n = 4000
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=n), index=make_index(n))
    s.iloc[500:2100] = 1.0  # 1600 minutes figées (plus de 24 h)
    s.iloc[3000:3100] = 2.0  # 100 minutes figées (moins de 24 h)
    df = pd.DataFrame({"s": s})

    masked, flags = mask_frozen(df, ["s"])

    assert flags["s"].iloc[500:2100].all()
    assert not flags["s"].iloc[3000:3100].any()
    assert flags["s"].sum() == 1600
    assert masked["s"].iloc[500:2100].isna().all()
    assert masked["s"].iloc[3000:3100].notna().all()


def test_mask_frozen_ne_considere_pas_les_nan_comme_figes():
    n = 4000
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=n), index=make_index(n))
    s.iloc[100:2000] = np.nan  # un long trou n'est pas une séquence figée
    df = pd.DataFrame({"s": s})

    _, flags = mask_frozen(df, ["s"])

    assert not flags["s"].any()


def test_fill_short_gaps_respecte_la_limite_et_la_causalite():
    n = 60
    idx = make_index(n)
    status = pd.Series("NORMAL", index=idx)
    s = pd.Series(np.arange(n, dtype=float), index=idx)
    s.iloc[0:2] = np.nan  # début de série : rien avant, donc rien à reporter
    s.iloc[10:13] = np.nan  # trou court : rempli
    s.iloc[20:30] = np.nan  # trou long : 5 points remplis, 5 restent NaN

    out = fill_short_gaps(pd.DataFrame({"s": s}), ["s"], status, limit=5)["s"]

    assert out.iloc[0:2].isna().all()
    assert (out.iloc[10:13] == 9.0).all()
    assert (out.iloc[20:25] == 19.0).all()
    assert out.iloc[25:30].isna().all()


def test_fill_short_gaps_ne_traverse_pas_un_changement_de_statut():
    idx = make_index(30)
    status = pd.Series(["NORMAL"] * 15 + ["RECOVERING"] * 15, index=idx)
    s = pd.Series(np.arange(30, dtype=float), index=idx)
    s.iloc[15:18] = np.nan  # trou juste après le changement de statut

    out = fill_short_gaps(pd.DataFrame({"s": s}), ["s"], status, limit=5)["s"]

    assert out.iloc[15:18].isna().all()


def test_clean_pump_de_bout_en_bout():
    n = 10_000
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "machine_status": "NORMAL",
            "ok": rng.normal(size=n),
            "mort": np.nan,
            "fige": rng.normal(size=n),
            "trop_de_nan": rng.normal(size=n),
        },
        index=make_index(n),
    )
    df.loc[df.index[200:1800], "fige"] = 3.0  # 1600 minutes figées (16 % du capteur)
    df.loc[df.index[:3000], "trop_de_nan"] = np.nan  # 30 % de NaN en début de série
    df.loc[df.index[1500:1503], "ok"] = np.nan  # trou court
    df.loc[df.index[50], "ok"] = 1e6  # valeur extrême : doit être conservée
    sensors = ["ok", "mort", "fige", "trop_de_nan"]

    res = clean_pump(df, sensors, dead_sensors=["mort"])

    # capteur mort retiré, aucune ligne perdue, statuts inchangés
    assert "mort" in res.dropped
    assert "mort" not in res.data.columns
    assert res.data.index.equals(df.index)
    assert (res.data["machine_status"] == "NORMAL").all()
    # séquence figée masquée, trou court rempli, valeur extrême conservée
    assert res.data["fige"].iloc[200:1800].isna().all()
    assert res.data["ok"].iloc[1500:1503].notna().all()
    assert res.data["ok"].iloc[50] == 1e6
    # sélection des variables
    assert "ok" in res.features
    assert "fige" in res.features
    assert "trop_de_nan" not in res.features
    assert "trop_de_nan" in res.data.columns
    # rapport
    assert res.report.loc["fige", "fige_masque_pct"] == pytest.approx(16.0)
    assert res.report.loc["trop_de_nan", "statut"].startswith("exclu")
