"""Phase 3 : segmentation des régimes de fonctionnement (Pump).

Le dataset Pump n'a pas de variable de consigne (contrairement aux ``op_1..3`` de
C-MAPSS) : les régimes sont donc appris sur les capteurs eux-mêmes, par clustering.

Chaîne, ajustée **uniquement sur la zone ``normal`` du pli d'entraînement** :
agrégation (médiane sur ``REGIME_RESAMPLE``) → imputation par la médiane →
z-score robuste borné → PCA → KMeans (k choisi par silhouette).

À la prédiction :
- un point trop loin du centre de son régime (au-delà du quantile ``envelope_q`` des
  distances d'entraînement) est classé ``HORS_REGIME`` : c'est déjà un signal d'anomalie ;
- un point avec plus de la moitié des capteurs manquants est classé ``MANQUANT`` ;
- ``confirm_regimes`` ne valide un changement qu'après ``min_duration`` (causal) ;
- ``transition_mask`` marque la zone qui suit chaque changement confirmé.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    RANDOM_STATE,
    REGIME_CLIP,
    REGIME_ENVELOPE_Q,
    REGIME_K_RANGE,
    REGIME_MIN_DURATION,
    REGIME_PCA_VARIANCE,
    REGIME_RESAMPLE,
    REGIME_TRANSITION,
)

HORS_REGIME = -1
MANQUANT = -2


def aggregate(df: pd.DataFrame, features: list[str], freq: str = REGIME_RESAMPLE) -> pd.DataFrame:
    """Médiane par pas de ``freq`` (robuste aux pointes isolées)."""
    return df[features].resample(freq).median()


@dataclass
class RegimeModel:
    n_regimes: int | None = None  # None : choisi par silhouette dans ``k_range``
    k_range: tuple[int, int] = REGIME_K_RANGE
    pca_variance: float = REGIME_PCA_VARIANCE
    clip: float = REGIME_CLIP
    envelope_q: float = REGIME_ENVELOPE_Q
    random_state: int = RANDOM_STATE
    silhouette_: dict[int, float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ apprentissage
    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X.fillna(self.median_) - self.median_) / self.iqr_
        return np.clip(Z.to_numpy(dtype=float), -self.clip, self.clip)

    def fit(self, X: pd.DataFrame) -> "RegimeModel":
        """``X`` : données agrégées de la zone d'entraînement uniquement."""
        X = X[X.notna().mean(axis=1) > 0.5]
        self.features_ = list(X.columns)
        self.median_ = X.median()
        iqr = X.quantile(0.75) - X.quantile(0.25)
        self.iqr_ = iqr.where(iqr > 0, X.std()).replace(0, 1.0)

        Z = self._scale(X)
        self.pca_ = PCA(n_components=self.pca_variance, random_state=self.random_state).fit(Z)
        P = self.pca_.transform(Z)

        if self.n_regimes is None:
            for k in range(self.k_range[0], self.k_range[1] + 1):
                labels = KMeans(k, n_init=10, random_state=self.random_state).fit_predict(P)
                self.silhouette_[k] = silhouette_score(
                    P, labels, sample_size=min(5000, len(P)), random_state=self.random_state
                )
            k = max(self.silhouette_, key=self.silhouette_.get)
        else:
            k = self.n_regimes
        self.k_ = k
        self.kmeans_ = KMeans(k, n_init=10, random_state=self.random_state).fit(P)

        # Les régimes sont renumérotés par taille décroissante : 0 = le plus fréquent.
        raw = self.kmeans_.labels_
        order = np.argsort(-np.bincount(raw, minlength=k))
        self._relabel = np.empty(k, dtype=int)
        self._relabel[order] = np.arange(k)

        dist = self._distances(P)[np.arange(len(P)), raw]
        self.envelope_ = np.array(
            [np.quantile(dist[raw == j], self.envelope_q) for j in range(k)]
        )
        return self

    # ------------------------------------------------------------------- prédiction
    def _distances(self, P: np.ndarray) -> np.ndarray:
        c = self.kmeans_.cluster_centers_
        return np.sqrt(((P[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Régime de chaque pas (0 = le plus fréquent), ou ``HORS_REGIME`` / ``MANQUANT``."""
        X = X[self.features_]
        out = pd.Series(MANQUANT, index=X.index, dtype="int64")
        ok = (X.notna().mean(axis=1) > 0.5).to_numpy()
        if not ok.any():
            return out
        P = self.pca_.transform(self._scale(X[ok]))
        d = self._distances(P)
        raw = d.argmin(axis=1)
        inside = d[np.arange(len(raw)), raw] <= self.envelope_[raw]
        out[ok] = np.where(inside, self._relabel[raw], HORS_REGIME)
        return out

    def profile(self, X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Médiane des z-scores robustes de chaque capteur, par régime."""
        Z = pd.DataFrame(self._scale(X[self.features_]), index=X.index, columns=self.features_)
        keep = labels >= 0
        return Z[keep].groupby(labels[keep]).median().T


# ------------------------------------------------------------------ post-traitement

def confirm_regimes(labels: pd.Series, min_duration: str = REGIME_MIN_DURATION) -> pd.Series:
    """Ne valide un changement de régime qu'après ``min_duration`` de persistance.

    Causal : tant qu'un nouveau régime n'a pas duré ``min_duration``, on garde le
    régime confirmé précédent (le délai de confirmation est le prix de la causalité).
    ``HORS_REGIME`` et ``MANQUANT`` ne sont jamais lissés : ils passent tels quels.
    """
    step = labels.index.to_series().diff().median()
    need = max(1, int(np.ceil(pd.Timedelta(min_duration) / step)))
    v = labels.to_numpy()
    out = v.copy()
    current, candidate, count = None, None, 0
    for i, x in enumerate(v):
        if x < 0:
            continue
        if current is None:
            current = x
        if x == current:
            candidate, count = None, 0
        else:
            if x == candidate:
                count += 1
            else:
                candidate, count = x, 1
            if count >= need:
                current, candidate, count = x, None, 0
        out[i] = current
    return pd.Series(out, index=labels.index, name=labels.name)


def transition_mask(labels: pd.Series, duration: str = REGIME_TRANSITION) -> pd.Series:
    """``True`` pendant ``duration`` après chaque changement entre deux régimes valides."""
    valid = labels.where(labels >= 0).ffill()
    change = valid.ne(valid.shift()) & valid.shift().notna()
    starts = labels.index[change.to_numpy()]
    mask = pd.Series(False, index=labels.index)
    for t in starts:
        mask[(labels.index >= t) & (labels.index < t + pd.Timedelta(duration))] = True
    return mask


# ---------------------------------------------------------------------- diagnostics

def regime_table(
    labels: pd.Series,
    zones: pd.Series,
    failure_times: list[pd.Timestamp],
) -> pd.DataFrame:
    """Par régime : part du temps, durée médiane d'un épisode, lien avec les pannes.

    ``jours_avant_panne_med`` : délai médian jusqu'à la panne suivante (zone normal).
    ``part_pre_panne`` : part des points du régime situés en fenêtre pré-panne.
    """
    step = labels.index.to_series().diff().median()
    ft = np.array(sorted(failure_times), dtype="datetime64[ns]")
    t = labels.index.to_numpy()
    pos = np.searchsorted(ft, t, side="right")
    nxt = np.where(pos < len(ft), ft[np.minimum(pos, len(ft) - 1)], np.datetime64("NaT", "ns"))
    days = pd.Series((nxt - t) / np.timedelta64(1, "D"), index=labels.index)

    runs = labels.ne(labels.shift()).cumsum()
    run_len = labels.groupby(runs).agg(["first", "size"])

    rows = {}
    for r in sorted(labels.unique()):
        sel = labels == r
        evaluated = sel & zones.isin(["normal", "pre_failure"])
        rows[r] = {
            "part_temps_pct": 100 * sel.mean(),
            "n_episodes": int((run_len["first"] == r).sum()),
            "duree_episode_med": run_len.loc[run_len["first"] == r, "size"].median() * step,
            "jours_avant_panne_med": days[sel & (zones == "normal")].median(),
            "part_pre_panne_pct": 100 * (zones[evaluated] == "pre_failure").mean()
            if evaluated.any() else np.nan,
        }
    out = pd.DataFrame(rows).T
    out.index.name = "regime"
    return out
