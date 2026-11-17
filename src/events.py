"""Pannes du dataset Pump et zones temporelles du protocole d'évaluation.

Une panne = une séquence contiguë de ``BROKEN``. Son instant ``t_f`` est le premier
point ``BROKEN`` ; la période ``RECOVERING`` qui suit est la remise en service.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import LOFO_PURGE, PRE_FAILURE_WINDOW, STATUS_BROKEN, STATUS_NORMAL, STATUS_RECOVERING


@dataclass(frozen=True)
class FailureEvent:
    event_id: int
    failure_time: pd.Timestamp
    recovery_start: pd.Timestamp | None
    recovery_end: pd.Timestamp | None

    @property
    def downtime(self) -> pd.Timedelta | None:
        if self.recovery_end is None:
            return None
        return self.recovery_end - self.failure_time


def _status_runs(status: pd.Series) -> pd.DataFrame:
    """Découpe une série d'états en séquences contiguës (état, début, fin)."""
    run_id = (status != status.shift()).cumsum()
    runs = status.groupby(run_id).agg(
        state="first",
        start=lambda s: s.index[0],
        end=lambda s: s.index[-1],
    )
    return runs.reset_index(drop=True)


def extract_failure_events(status: pd.Series) -> list[FailureEvent]:
    """Liste des pannes, à partir de ``machine_status`` indexé par timestamp."""
    runs = _status_runs(status.astype(str))
    events: list[FailureEvent] = []
    for i, run in runs.iterrows():
        if run.state != STATUS_BROKEN:
            continue
        nxt = runs.iloc[i + 1] if i + 1 < len(runs) else None
        recovering = nxt is not None and nxt.state == STATUS_RECOVERING
        events.append(
            FailureEvent(
                event_id=len(events) + 1,
                failure_time=run.start,
                recovery_start=nxt.start if recovering else None,
                recovery_end=nxt.end if recovering else None,
            )
        )
    return events


def events_to_frame(events: list[FailureEvent]) -> pd.DataFrame:
    """Tableau lisible des pannes (une ligne par panne)."""
    rows = [
        {
            "event_id": e.event_id,
            "failure_time": e.failure_time,
            "recovery_start": e.recovery_start,
            "recovery_end": e.recovery_end,
            "downtime": e.downtime,
        }
        for e in events
    ]
    df = pd.DataFrame(rows).set_index("event_id")
    df["time_since_previous_failure"] = df["failure_time"].diff()
    return df


def pre_failure_mask(
    index: pd.DatetimeIndex,
    events: list[FailureEvent],
    window: str | pd.Timedelta = PRE_FAILURE_WINDOW,
) -> pd.Series:
    """``True`` sur ``[t_f - window, t_f[`` pour chaque panne (zone positive du protocole).

    La fenêtre est tronquée à la fin de la remise en service de la panne précédente :
    une machine en ``RECOVERING`` n'est pas « en pré-panne ».
    """
    window = pd.Timedelta(window)
    mask = pd.Series(False, index=index)
    previous_recovery_end = None
    for e in events:
        start = e.failure_time - window
        if previous_recovery_end is not None:
            start = max(start, previous_recovery_end + pd.Timedelta(1, "ns"))
        mask[(index >= start) & (index < e.failure_time)] = True
        previous_recovery_end = e.recovery_end or e.failure_time
    return mask


def time_zones(
    status: pd.Series,
    events: list[FailureEvent],
    window: str | pd.Timedelta = PRE_FAILURE_WINDOW,
) -> pd.Series:
    """Zone du protocole pour chaque instant.

    - ``pre_failure`` : positif (évalué, exclu de l'entraînement)
    - ``normal``      : négatif (évalué, utilisable à l'entraînement)
    - ``excluded``    : BROKEN / RECOVERING (ni entraîné, ni évalué)
    - ``censored``    : dernière ``window`` des données, sans panne connue après
                        (étiquette inconnue → ni entraîné, ni évalué)
    """
    window = pd.Timedelta(window)
    zones = pd.Series("excluded", index=status.index, dtype=object)
    zones[status.astype(str) == STATUS_NORMAL] = "normal"
    zones[pre_failure_mask(status.index, events, window)] = "pre_failure"

    last_failure = events[-1].failure_time if events else status.index[0]
    censored = (status.index > status.index[-1] - window) & (status.index > last_failure)
    zones[censored & (zones == "normal")] = "censored"
    return zones


def failure_cycles(index: pd.DatetimeIndex, events: list[FailureEvent]) -> pd.Series:
    """Numéro de cycle du protocole (section 4) pour chaque instant.

    Cycle k = ``]fin de remise en service k-1, fin de remise en service k]`` ; le premier
    cycle commence au début des données. Le segment après la dernière remise en service
    reçoit le numéro ``len(events) + 1`` : c'est le cycle sans panne, toujours à l'entraînement.
    """
    cycles = pd.Series(len(events) + 1, index=index, dtype="int64")
    start = None
    for e in events:
        end = e.recovery_end or e.failure_time
        sel = index <= end if start is None else (index > start) & (index <= end)
        cycles[sel] = e.event_id
        start = end
    return cycles


def lofo_folds(
    zones: pd.Series,
    cycles: pd.Series,
    events: list[FailureEvent],
    purge: str | pd.Timedelta = LOFO_PURGE,
    chronological: bool = False,
):
    """Plis leave-one-failure-out du protocole : ``(k, masque_train, masque_test)``.

    - test  : le cycle k entier (les zones disent ensuite quoi évaluer) ;
    - train : zone ``normal`` des autres cycles, moins ``purge`` de part et d'autre du cycle k.
    Avec ``chronological=True`` (contrôle de la section 4), seuls les cycles < k entraînent,
    et le premier pli est sauté.
    """
    purge = pd.Timedelta(purge)
    index = zones.index
    for e in events:
        k = e.event_id
        test = cycles == k
        start, end = index[test.to_numpy()].min(), index[test.to_numpy()].max()
        near = (index >= start - purge) & (index <= end + purge)
        train = (zones == "normal") & ~test & ~near
        if chronological:
            if k == 1:
                continue
            train &= cycles < k
        yield k, train, test
