import numpy as np
import pandas as pd

from src.regimes import (
    HORS_REGIME,
    MANQUANT,
    RegimeModel,
    confirm_regimes,
    regime_table,
    transition_mask,
)


def _two_regimes(n=600, seed=0):
    """Deux régimes nets (niveaux opposés sur 4 capteurs), en blocs de 100 pas."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-04-01", periods=n, freq="10min")
    block = (np.arange(n) // 100) % 2
    X = rng.normal(scale=0.3, size=(n, 4)) + np.where(block[:, None] == 1, 5.0, 0.0)
    return pd.DataFrame(X, index=idx, columns=[f"sensor_0{i}" for i in range(4)]), block


def test_regime_model_retrouve_deux_regimes():
    X, block = _two_regimes()
    model = RegimeModel(k_range=(2, 4)).fit(X)
    assert model.k_ == 2
    labels = model.predict(X)
    valid = labels >= 0
    # Les étiquettes recouvrent les blocs, à une permutation près.
    agree = (labels[valid].to_numpy() == block[valid.to_numpy()]).mean()
    assert max(agree, 1 - agree) > 0.98


def test_regime_model_hors_regime_et_manquant():
    X, _ = _two_regimes()
    model = RegimeModel(n_regimes=2).fit(X)
    Y = X.iloc[:3].copy()
    Y.iloc[0] = 50.0          # très loin des deux régimes
    Y.iloc[1] = np.nan        # aucun capteur
    labels = model.predict(Y)
    assert labels.iloc[0] == HORS_REGIME
    assert labels.iloc[1] == MANQUANT
    assert labels.iloc[2] >= 0


def test_regime_model_numerote_par_taille():
    X, _ = _two_regimes()
    X = X.iloc[:300]  # blocs 0, 1, 0 : le régime « bas » est majoritaire
    model = RegimeModel(n_regimes=2).fit(X)
    labels = model.predict(X)
    assert (labels == 0).sum() > (labels == 1).sum()


def test_confirm_regimes_est_causal_et_filtre_les_rafales():
    idx = pd.date_range("2018-04-01", periods=12, freq="10min")
    raw = pd.Series([0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], index=idx)
    out = confirm_regimes(raw, "30min")  # 3 pas pour confirmer
    assert out.iloc[2] == 0              # rafale isolée ignorée
    assert (out.iloc[5:7] == 0).all()    # pas encore confirmé
    assert (out.iloc[7:] == 1).all()     # confirmé au 3e pas
    # Le passé ne dépend pas du futur.
    assert out.iloc[:7].equals(confirm_regimes(raw.iloc[:7], "30min"))


def test_confirm_regimes_laisse_passer_les_codes_speciaux():
    idx = pd.date_range("2018-04-01", periods=4, freq="10min")
    raw = pd.Series([0, HORS_REGIME, MANQUANT, 0], index=idx)
    assert confirm_regimes(raw, "30min").tolist() == [0, HORS_REGIME, MANQUANT, 0]


def test_transition_mask():
    idx = pd.date_range("2018-04-01", periods=8, freq="10min")
    labels = pd.Series([0, 0, 0, 1, 1, 1, 1, 1], index=idx)
    mask = transition_mask(labels, "20min")
    assert mask.tolist() == [False, False, False, True, True, False, False, False]


def test_regime_table():
    idx = pd.date_range("2018-04-01", periods=6, freq="1D")
    labels = pd.Series([0, 0, 1, 1, 0, 0], index=idx)
    zones = pd.Series(["normal", "normal", "pre_failure", "pre_failure", "normal", "normal"], index=idx)
    table = regime_table(labels, zones, [pd.Timestamp("2018-04-05")])
    assert table.loc[1, "part_pre_panne_pct"] == 100
    assert table.loc[0, "n_episodes"] == 2
    assert table.loc[0, "jours_avant_panne_med"] == 3.5  # 4 j et 3 j avant la panne
