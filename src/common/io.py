"""Loading helpers for processed data. Every metric module should load through here."""
import pandas as pd
import pyarrow.dataset as ds

from .config import PROC, TRACK_DIR


def plays() -> pd.DataFrame:
    """One row per play: context, split, snap/throw/end frames, target, pressure."""
    return pd.read_parquet(PROC / "plays.parquet")


def player_plays() -> pd.DataFrame:
    """One row per player-play: PFF scouting joined to player info + alignment at snap."""
    return pd.read_parquet(PROC / "player_plays.parquet")


def te_plays() -> pd.DataFrame:
    """One row per TE-play with assignment category, release, target/catch flags."""
    return pd.read_parquet(PROC / "te_plays.parquet")


def receiver_frames(columns=None, filters=None) -> pd.DataFrame:
    """Per-frame features for every route runner (all positions), snap -> end of dropback."""
    return pd.read_parquet(PROC / "receiver_frames.parquet", columns=columns, filters=filters)


def routes() -> pd.DataFrame:
    """One row per route-runner-play (all positions) with release + throw-frame summary."""
    return pd.read_parquet(PROC / "routes.parquet")


def tracking(columns=None, filters=None) -> pd.DataFrame:
    """Standardized tracking (offense moves +x), frames from snap to end of tracking.

    filters example: [("gameId", "in", [2021090900])]
    """
    return ds.dataset(str(TRACK_DIR), format="parquet").to_table(
        columns=columns, filter=_to_expr(filters)).to_pandas()


def _to_expr(filters):
    if not filters:
        return None
    expr = None
    for col, op, val in filters:
        f = ds.field(col)
        ops = {"==": lambda: f == val, "in": lambda: f.isin(list(val)), ">=": lambda: f >= val,
               "<=": lambda: f <= val, ">": lambda: f > val, "<": lambda: f < val}
        e = ops[op]()
        expr = e if expr is None else expr & e
    return expr
