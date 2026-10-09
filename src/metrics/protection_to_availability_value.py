"""Matched-control Protection-to-Availability Value (PAV) for tight ends.

PAV estimates whether a TE chip-and-release assignment extends the time before
a labeled pass rusher reaches the QB threat zone, then weights that time lift by
the TE's receiving availability at the end of the dropback. The expected
no-chip protection window comes from nearest context-matched immediate-release
TE routes.

Run from the project root:
    .venv/bin/python -m src.metrics.protection_to_availability_value
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import io, stats
from src.common.config import (
    FIELD_W,
    SEED,
    TEST_WEEKS,
    TRAIN_WEEKS,
    VIABLE_DEPTH,
    VIABLE_MIN_LANE,
    VIABLE_MIN_SEP,
    VIABLE_SIDELINE_BUFFER,
)

SLUG = "protection_to_availability_value"
OUT_DIR = Path("output") / SLUG
KEY = ["gameId", "playId", "nflId"]

THREAT_RADIUS_YARDS = 2.0
THREAT_MIN_TIME_SECONDS = 0.5
N_MATCHES = 8
MIN_CHIP_PLAYS = 8

CAT_FEATURES = [
    "alignment",
    "offenseFormation",
    "personnelO",
    "personnelD",
    "dropBackType",
    "pff_passCoverageType",
    "pff_passCoverage",
]
NUM_FEATURES = [
    "down",
    "yardsToGo",
    "defendersInBox",
    "pff_playAction",
    "n_rushers",
    "n_pass_blockers",
    "n_routes",
    "n_te_on_field",
    "x_rel_snap",
    "lat_from_ball",
]
MATCH_FEATURES = NUM_FEATURES + CAT_FEATURES


def load_metric_frame() -> pd.DataFrame:
    """Load TE route assignments and only pre-treatment play context."""
    te = io.te_plays()
    te = te.loc[te.assignment.isin(["route", "chip_release"])].copy()
    te = te.drop(columns=["week", "split", "is_train"], errors="ignore")
    plays = io.plays()
    play_columns = [
        "gameId",
        "playId",
        "week",
        "split",
        "is_train",
        "down",
        "yardsToGo",
        "defendersInBox",
        "offenseFormation",
        "personnelO",
        "personnelD",
        "dropBackType",
        "pff_playAction",
        "pff_passCoverageType",
        "pff_passCoverage",
        "n_rushers",
        "n_pass_blockers",
        "n_routes",
        "n_te_on_field",
        "pressure",
        "time_to_end",
    ]
    routes = io.routes()[KEY + ["t_end", "x_rel_end", "y_end", "sep_end", "lane_end", "released_flag"]]
    frame = te.merge(plays[play_columns], on=["gameId", "playId"], how="left").merge(
        routes, on=KEY, how="left", suffixes=("", "_route")
    )
    frame["is_chip"] = frame.assignment.eq("chip_release")
    frame["is_immediate_control"] = frame.assignment.eq("route") & frame.release_t.notna()
    frame["has_route_tracking"] = frame.t_end.notna() & frame.release_t.notna()
    frame["lat_from_ball"] = frame["lat_from_ball"].astype(float)
    frame["pff_playAction"] = frame["pff_playAction"].fillna(0).astype(float)
    for column in NUM_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame[column].median())
    for column in CAT_FEATURES:
        frame[column] = frame[column].fillna("Unknown").astype(str)
    return frame


def protection_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Find first closing pass-rusher entry into the QB threat zone by play."""
    wanted = frame[["gameId", "playId"]].drop_duplicates()
    outputs: list[pd.DataFrame] = []

    for game_id, game_plays in wanted.groupby("gameId"):
        play_ids = game_plays.playId.tolist()
        tracking = io.tracking(
            columns=["gameId", "playId", "frameId", "t", "nflId", "pff_role", "x", "y"],
            filters=[("gameId", "==", int(game_id))],
        )
        tracking = tracking.loc[tracking.playId.isin(play_ids)]
        qb = tracking.loc[tracking.pff_role.eq("Pass"), ["playId", "frameId", "t", "x", "y"]].rename(
            columns={"x": "qb_x", "y": "qb_y"}
        )
        rush = tracking.loc[
            tracking.pff_role.eq("Pass Rush"), ["playId", "frameId", "t", "nflId", "x", "y"]
        ]
        pairs = rush.merge(qb, on=["playId", "frameId", "t"], how="inner")
        if pairs.empty:
            continue
        pairs["distance_to_qb"] = np.hypot(pairs.x - pairs.qb_x, pairs.y - pairs.qb_y)
        nearest = pairs.groupby(["playId", "frameId", "t"], as_index=False).distance_to_qb.min()
        nearest["closing"] = nearest.groupby("playId").distance_to_qb.diff().lt(0)
        threats = nearest.loc[
            nearest.t.ge(THREAT_MIN_TIME_SECONDS)
            & nearest.distance_to_qb.le(THREAT_RADIUS_YARDS)
            & nearest.closing
        ]
        first = threats.groupby("playId", as_index=False).t.min().rename(columns={"t": "threat_time"})
        first.insert(0, "gameId", int(game_id))
        outputs.append(first)

    windows = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(
        columns=["gameId", "playId", "threat_time"]
    )
    return frame.merge(windows, on=["gameId", "playId"], how="left")


def receiving_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Score 0-1 availability at the throw/end-of-dropback frame."""
    depth_ok = frame.x_rel_end.between(*VIABLE_DEPTH)
    sideline_distance = np.minimum(frame.y_end, FIELD_W - frame.y_end)
    frame["availability_separation"] = (frame.sep_end / VIABLE_MIN_SEP).clip(0, 1)
    frame["availability_lane"] = (frame.lane_end / VIABLE_MIN_LANE).clip(0, 1)
    frame["availability_depth"] = depth_ok.astype(float)
    frame["availability_sideline"] = (sideline_distance >= VIABLE_SIDELINE_BUFFER).astype(float)
    components = [
        "availability_separation",
        "availability_lane",
        "availability_depth",
        "availability_sideline",
    ]
    frame["availability_score"] = frame[components].mean(axis=1)
    frame.loc[~frame.has_route_tracking, "availability_score"] = 0.0
    return frame


def make_matcher() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), NUM_FEATURES),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                    ]
                ),
            )
        ]
    )


def nearest_control_expectation(
    train_controls: pd.DataFrame, evaluation: pd.DataFrame
) -> pd.DataFrame:
    """Return distance-weighted control outcomes for each evaluation row."""
    if train_controls.empty or evaluation.empty:
        return pd.DataFrame(index=evaluation.index)

    matcher = make_matcher().fit(train_controls)
    control_features = matcher.transform(train_controls)
    evaluation_features = matcher.transform(evaluation)
    neighbors = NearestNeighbors(n_neighbors=min(N_MATCHES, len(train_controls))).fit(control_features)
    distances, indices = neighbors.kneighbors(evaluation_features)
    weights = 1 / np.maximum(distances, 0.05)
    control_windows = train_controls["protection_window_seconds"].to_numpy()[indices]
    control_pressures = train_controls["pressure"].to_numpy()[indices]
    result = pd.DataFrame(index=evaluation.index)
    result["expected_protection_window_seconds"] = np.average(control_windows, axis=1, weights=weights)
    result["expected_pressure_rate"] = np.average(control_pressures, axis=1, weights=weights)
    result["mean_match_distance"] = distances.mean(axis=1)
    result["matched_control_count"] = len(train_controls)
    return result


def cross_fit_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-fit train rows by game and use all train controls for held-out weeks."""
    output = pd.DataFrame(index=frame.index)
    train = frame.loc[frame.is_train].copy()
    controls = train.loc[train.is_immediate_control].copy()
    groups = train.gameId
    for train_index, validation_index in GroupKFold(n_splits=5).split(train, groups=groups):
        fold_train = train.iloc[train_index]
        fold_controls = fold_train.loc[fold_train.is_immediate_control]
        fold_evaluation = train.iloc[validation_index]
        matched = nearest_control_expectation(fold_controls, fold_evaluation)
        output.loc[fold_evaluation.index, matched.columns] = matched

    test = frame.loc[~frame.is_train]
    output.loc[test.index, nearest_control_expectation(controls, test).columns] = nearest_control_expectation(
        controls, test
    )
    return pd.concat([frame, output], axis=1)


def regression_report(actual: pd.Series, predicted: pd.Series, baseline: float) -> dict[str, float]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted, squared=False)),
        "baseline_mae": float(mean_absolute_error(actual, np.full(len(actual), baseline))),
        "baseline_rmse": float(mean_squared_error(actual, np.full(len(actual), baseline), squared=False)),
    }


def player_table(chips: pd.DataFrame) -> pd.DataFrame:
    pooled_sd = float(chips.pav.std()) if len(chips) > 1 else 0.0
    result = chips.groupby(["nflId", "displayName"], as_index=False).agg(
        n=("pav", "size"),
        value=("pav", "mean"),
        mean_time_lift_seconds=("protection_time_lift_seconds", "mean"),
        mean_availability_score=("availability_score", "mean"),
        mean_match_distance=("mean_match_distance", "mean"),
        observed_pressure_rate=("pressure", "mean"),
        expected_pressure_rate=("expected_pressure_rate", "mean"),
    )
    result["value_shrunk"] = stats.eb_shrink_mean(
        result.value, result.n, np.full(len(result), pooled_sd)
    )
    standard_error = pooled_sd / np.sqrt(result.n)
    result["lo"] = result.value - 1.96 * standard_error
    result["hi"] = result.value + 1.96 * standard_error
    result["higher_is_better"] = True
    result["qualifies_top"] = result.n.ge(MIN_CHIP_PLAYS)
    return result.sort_values(["qualifies_top", "value_shrunk"], ascending=[False, False])


def reliability(chips: pd.DataFrame) -> dict[str, object]:
    counts = chips.groupby(["nflId", "is_train"]).pav.count().unstack(fill_value=0)
    eligible = counts.index[
        (counts.get(True, 0) >= 3) & (counts.get(False, 0) >= 3)
    ]
    if len(eligible) < 5:
        return {"spearman": None, "n_players": int(len(eligible)), "minimum_plays_per_split": 3}
    train_values = chips.loc[chips.is_train & chips.nflId.isin(eligible)].groupby("nflId").pav.mean()
    test_values = chips.loc[~chips.is_train & chips.nflId.isin(eligible)].groupby("nflId").pav.mean()
    return {
        "spearman": float(train_values.corr(test_values, method="spearman")),
        "n_players": int(len(eligible)),
        "minimum_plays_per_split": 3,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = receiving_availability(protection_windows(load_metric_frame()))
    frame["protection_window_seconds"] = frame.threat_time.fillna(frame.time_to_end)
    frame = cross_fit_matches(frame)

    analysis = frame.loc[
        frame.is_chip
        & frame.has_route_tracking
        & frame.expected_protection_window_seconds.notna()
    ].copy()
    analysis["protection_time_lift_seconds"] = (
        analysis.protection_window_seconds - analysis.expected_protection_window_seconds
    )
    analysis["pav"] = analysis.protection_time_lift_seconds * analysis.availability_score

    test_controls = frame.loc[(~frame.is_train) & frame.is_immediate_control]
    baseline = float(frame.loc[frame.is_train & frame.is_immediate_control, "protection_window_seconds"].mean())
    validation = regression_report(
        test_controls.protection_window_seconds,
        test_controls.expected_protection_window_seconds,
        baseline,
    )
    chip_test = analysis.loc[~analysis.is_train]
    matched_control_pressure = float(chip_test.expected_pressure_rate.mean()) if len(chip_test) else np.nan
    observed_chip_pressure = float(chip_test.pressure.mean()) if len(chip_test) else np.nan

    players = player_table(analysis)
    player_columns = [
        "nflId",
        "displayName",
        "n",
        "value",
        "value_shrunk",
        "lo",
        "hi",
        "higher_is_better",
        "qualifies_top",
        "mean_time_lift_seconds",
        "mean_availability_score",
        "mean_match_distance",
        "observed_pressure_rate",
        "expected_pressure_rate",
    ]
    players[player_columns].to_csv(OUT_DIR / "players.csv", index=False, float_format="%.4f")

    play_columns = [
        "gameId",
        "playId",
        "nflId",
        "displayName",
        "week",
        "is_train",
        "assignment",
        "alignment",
        "release_t",
        "protection_window_seconds",
        "threat_time",
        "expected_protection_window_seconds",
        "protection_time_lift_seconds",
        "availability_score",
        "availability_separation",
        "availability_lane",
        "availability_depth",
        "availability_sideline",
        "pav",
        "pressure",
        "expected_pressure_rate",
        "mean_match_distance",
    ]
    analysis.sort_values(KEY)[play_columns].to_csv(OUT_DIR / "plays.csv", index=False, float_format="%.4f")

    summary = {
        "metric": "Protection-to-Availability Value (PAV)",
        "definition": (
            "For a TE chip-and-release play, PAV = (observed protection window - "
            "matched immediate-release control expectation) * receiving availability score."
        ),
        "treatment": "TE PFF assignment chip_release (Pass Route with CH or SR block type).",
        "control": "TE PFF route assignment without a CH or SR chip label and with tracking-confirmed release.",
        "protection_window": (
            "Seconds from snap to the first labeled pass rusher entering a "
            f"{THREAT_RADIUS_YARDS}-yard, closing QB threat zone after {THREAT_MIN_TIME_SECONDS} seconds; "
            "time_to_end when no threat occurs."
        ),
        "availability": {
            "definition": "Mean of separation, passing-lane, viable-depth, and sideline components at end of dropback.",
            "thresholds": {
                "separation_yards": VIABLE_MIN_SEP,
                "lane_clearance_yards": VIABLE_MIN_LANE,
                "viable_depth_range": list(VIABLE_DEPTH),
                "sideline_buffer_yards": VIABLE_SIDELINE_BUFFER,
            },
        },
        "matching": {
            "method": "distance-weighted nearest-neighbor matching on immediate-release TE route controls",
            "n_neighbors": N_MATCHES,
            "features": MATCH_FEATURES,
            "train_weeks": TRAIN_WEEKS,
            "test_weeks": TEST_WEEKS,
            "cross_fit": "5-fold GroupKFold by game for train rows; all train controls for test rows",
        },
        "sample_sizes": {
            "te_route_or_chip_assignments": int(len(frame)),
            "immediate_release_controls": int(frame.is_immediate_control.sum()),
            "chip_assignments": int(frame.is_chip.sum()),
            "scored_chip_assignments": int(len(analysis)),
            "scored_chip_train": int(analysis.is_train.sum()),
            "scored_chip_test": int((~analysis.is_train).sum()),
        },
        "test_control_window_validation": validation,
        "test_chip_pressure_validation": {
            "n": int(len(chip_test)),
            "observed_pressure_rate": observed_chip_pressure,
            "matched_control_expected_pressure_rate": matched_control_pressure,
            "difference": None
            if pd.isna(observed_chip_pressure) or pd.isna(matched_control_pressure)
            else observed_chip_pressure - matched_control_pressure,
        },
        "split_half_reliability": reliability(analysis),
        "caveats": [
            "This is matched observational evidence, not a randomized causal effect.",
            "The source data has no ball-flight or catch-point tracking; availability is measured at end of dropback.",
            "PFF chip labels and role assignments define treatment and may contain classification error.",
            "Player leaderboards require at least eight scored chip-release plays.",
        ],
        "seed": SEED,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(analysis)} scored chip-release plays and {len(players)} TE rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
