"""Figures réutilisables (EDA, chronologie des pannes)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES, PRE_FAILURE_WINDOW, STATUS_NORMAL
from src.events import FailureEvent

STATUS_COLORS = {"ok": "#4C78A8", "suspect": "#F58518", "mort": "#E45756"}
FAILURE_COLOR = "#E45756"
RECOVERY_COLOR = "#F58518"
PRE_FAILURE_COLOR = "#9D9D9D"


def save_fig(fig: plt.Figure, name: str, directory: Path = FIGURES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def mark_events(
    ax: plt.Axes,
    events: list[FailureEvent],
    window: str | pd.Timedelta | None = PRE_FAILURE_WINDOW,
) -> None:
    """Pannes (trait rouge), remises en service (orange), fenêtres pré-panne (gris)."""
    for e in events:
        if window is not None:
            ax.axvspan(e.failure_time - pd.Timedelta(window), e.failure_time,
                       color=PRE_FAILURE_COLOR, alpha=0.25, lw=0)
        if e.recovery_start is not None:
            ax.axvspan(e.recovery_start, e.recovery_end, color=RECOVERY_COLOR, alpha=0.25, lw=0)
        ax.axvline(e.failure_time, color=FAILURE_COLOR, lw=1)


def plot_quality_overview(classified: pd.DataFrame) -> plt.Figure:
    """Part de NaN et plus longue séquence figée par capteur, colorées par statut."""
    colors = classified["status"].map(STATUS_COLORS)
    x = np.arange(len(classified))
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].bar(x, classified["nan_pct"], color=colors)
    axes[0].set_ylabel("NaN (%)")
    axes[1].bar(x, classified["longest_frozen"].dt.total_seconds() / 3600, color=colors)
    axes[1].set_ylabel("Plus longue séquence figée (h)")
    axes[1].set_yscale("symlog")
    axes[1].set_xticks(x, classified.index, rotation=90, fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLORS.values()]
    axes[0].legend(handles, STATUS_COLORS.keys(), loc="upper right")
    fig.tight_layout()
    return fig


def plot_failure_timeline(
    df: pd.DataFrame,
    sensors: list[str],
    events: list[FailureEvent],
    resample: str = "1h",
    window: str | pd.Timedelta = PRE_FAILURE_WINDOW,
) -> plt.Figure:
    """Un panneau par capteur sur toute la période, pannes superposées.

    Les signaux sont agrégés (moyenne sur ``resample``) pour rester lisibles.
    """
    data = df[sensors].resample(resample).mean()
    fig, axes = plt.subplots(len(sensors), 1, figsize=(14, 0.9 * len(sensors) + 1),
                             sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], sensors):
        ax.plot(data.index, data[name], lw=0.6, color="#333333")
        mark_events(ax, events, window)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(
        f"Signaux (moyenne {resample}) — panne en rouge, remise en service en orange, "
        f"fenêtre pré-panne {window} en gris"
    )
    fig.tight_layout()
    return fig


def robust_zscore(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """(x - médiane) / IQR, calculés sur ``reference`` (typiquement les périodes NORMAL)."""
    med = reference.median()
    iqr = (reference.quantile(0.75) - reference.quantile(0.25)).replace(0, np.nan)
    return (df - med) / iqr


def plot_failure_heatmaps(
    df: pd.DataFrame,
    sensors: list[str],
    events: list[FailureEvent],
    status_col: str = "machine_status",
    before: str = "72h",
    after: str = "12h",
    resample: str = "10min",
    clip: float = 5.0,
) -> plt.Figure:
    """Pour chaque panne, carte capteurs × temps des z-scores robustes autour de t_f.

    Référence du z-score : toutes les périodes NORMAL. Permet de repérer
    visuellement quels capteurs s'écartent avant la panne, sans présélection.
    """
    reference = df.loc[df[status_col].astype(str) == STATUS_NORMAL, sensors]
    fig, axes = plt.subplots(len(events), 1, figsize=(14, 0.13 * len(sensors) * len(events) + 1),
                             squeeze=False)
    image = None
    for ax, e in zip(axes[:, 0], events):
        start, end = e.failure_time - pd.Timedelta(before), e.failure_time + pd.Timedelta(after)
        z = robust_zscore(df.loc[start:end, sensors].resample(resample).mean(), reference)
        hours = (z.index - e.failure_time) / pd.Timedelta("1h")
        values = np.ma.masked_invalid(z.T.clip(-clip, clip).to_numpy())
        image = ax.pcolormesh(hours, np.arange(len(sensors)), values,
                              cmap="RdBu_r", vmin=-clip, vmax=clip, shading="nearest")
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(np.arange(len(sensors)), sensors, fontsize=5)
        ax.set_title(f"Panne {e.event_id} — {e.failure_time:%Y-%m-%d %H:%M}", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("Heures relatives à la panne (t_f = 0)")
    fig.colorbar(image, ax=axes[:, 0], shrink=0.4, label="z-score robuste (borné)")
    return fig


# --------------------------------------------------------------------- phase 3

REGIME_COLORS = {-2: "#bdbdbd", -1: "#000000", 0: "#4c78a8", 1: "#e45756", 2: "#54a24b",
                 3: "#b279a2", 4: "#f58518", 5: "#72b7b2"}
REGIME_NAMES = {-2: "manquant", -1: "hors régime"}


def plot_silhouette(scores: dict[int, float], chosen: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ks = sorted(scores)
    ax.plot(ks, [scores[k] for k in ks], marker="o", color="#4c78a8")
    ax.axvline(chosen, color="grey", ls="--", lw=1)
    ax.set_xlabel("nombre de régimes k")
    ax.set_ylabel("silhouette")
    ax.set_title(f"Choix de k (retenu : {chosen})")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_regime_profile(profile: pd.DataFrame, clip: float = 10.0) -> plt.Figure:
    """Carte des z-scores robustes médians, capteur × régime."""
    fig, ax = plt.subplots(figsize=(3 + 1.2 * profile.shape[1], 0.22 * len(profile) + 1.5))
    im = ax.imshow(profile.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-clip, vmax=clip)
    ax.set_yticks(range(len(profile)), profile.index, fontsize=7)
    ax.set_xticks(range(profile.shape[1]), [f"régime {c}" for c in profile.columns])
    fig.colorbar(im, ax=ax, label="z-score robuste médian (borné)")
    ax.set_title("Profil des régimes")
    return fig


def plot_regime_timeline(
    labels: pd.Series,
    events: list[FailureEvent],
    signals: pd.DataFrame | None = None,
) -> plt.Figure:
    """Bande des régimes dans le temps, pannes en rouge, signaux optionnels en dessous."""
    n = 1 + (0 if signals is None else signals.shape[1])
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.2 + 1.3 * n), sharex=True,
                             gridspec_kw={"height_ratios": [1] + [1.4] * (n - 1)})
    axes = np.atleast_1d(axes)
    ax = axes[0]
    step = labels.index.to_series().diff().median()
    runs = labels.ne(labels.shift()).cumsum()
    seen = set()
    for _, run in labels.groupby(runs):
        r = int(run.iloc[0])
        ax.axvspan(run.index[0], run.index[-1] + step, color=REGIME_COLORS.get(r, "#999"), lw=0,
                   label=None if r in seen else REGIME_NAMES.get(r, f"régime {r}"))
        seen.add(r)
    handles, names = ax.get_legend_handles_labels()
    order = np.argsort(names)
    ax.legend([handles[i] for i in order], [names[i] for i in order],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_yticks([])
    ax.set_title("Régimes (confirmés) et pannes")
    for i, col in enumerate([] if signals is None else signals.columns, start=1):
        axes[i].plot(signals.index, signals[col], lw=0.6, color="#333")
        axes[i].set_ylabel(col, fontsize=8)
    for a in axes:
        mark_events(a, events, window=None)
        a.spines[["top", "right"]].set_visible(False)
    return fig
