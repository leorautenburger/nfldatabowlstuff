#!/usr/bin/env python3
"""Compute descriptive tight-end Protection-to-Availability inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FRAMES_PER_SECOND = 10
RELEASE_DISPLACEMENT_YARDS = 1.5
RELEASE_SPEED_YARDS_PER_SECOND = 2.0
OPEN_SEPARATION_YARDS = 3.0
THREAT_RADIUS_YARDS = 2.0
MIN_THREAT_DELAY_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a descriptive tight-end chip-and-release dual-threat "
            "window from NFL Big Data Bowl tracking files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing players.csv, pffScoutingData.csv, and tracking/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/te_chip_release_pav.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--game-id",
        type=int,
        action="append",
        help="Optional game ID. Repeat to process more than one game.",
    )
    return parser.parse_args()


def read_candidates(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(data_dir / "players.csv", usecols=["nflId", "displayName", "officialPosition"])
    scouting = pd.read_csv(data_dir / "pffScoutingData.csv")

    tight_ends = players.loc[
        players["officialPosition"].eq("TE"), ["nflId", "displayName"]
    ]
    chip_routes = scouting.loc[
        scouting["pff_blockType"].eq("CH") & scouting["pff_role"].eq("Pass Route"),
        ["gameId", "playId", "nflId", "pff_nflIdBlockedPlayer"],
    ].merge(tight_ends, on="nflId", how="inner")
    passers = scouting.loc[
        scouting["pff_role"].eq("Pass"), ["gameId", "playId", "nflId"]
    ].rename(columns={"nflId": "qbNflId"})
    pass_rushers = scouting.loc[
        scouting["pff_role"].eq("Pass Rush"), ["gameId", "playId", "nflId"]
    ]
    return chip_routes, passers, pass_rushers


def first_frame_after(mask: pd.Series, frames: pd.Series) -> int | None:
    matching_frames = frames.loc[mask]
    return None if matching_frames.empty else int(matching_frames.iloc[0])


def sustained_release_frame(te_frames: pd.DataFrame, snap_frame: int) -> int | None:
    te_frames = te_frames.loc[te_frames["frameId"] >= snap_frame].copy()
    if te_frames.empty:
        return None

    snap_row = te_frames.loc[te_frames["frameId"].eq(snap_frame)]
    if snap_row.empty:
        return None

    snap_x, snap_y = snap_row.iloc[0][["x", "y"]]
    displacement = ((te_frames["x"] - snap_x) ** 2 + (te_frames["y"] - snap_y) ** 2) ** 0.5
    is_releasing = displacement.ge(RELEASE_DISPLACEMENT_YARDS) & te_frames["s"].ge(
        RELEASE_SPEED_YARDS_PER_SECOND
    )
    sustained = is_releasing.rolling(3, min_periods=3).sum().eq(3)
    return first_frame_after(sustained, te_frames["frameId"])


def analyze_play(
    play_tracking: pd.DataFrame,
    candidate: pd.Series,
    qb_id: int,
    pass_rusher_ids: set[int],
) -> dict[str, object] | None:
    snap_frame = first_frame_after(
        play_tracking["event"].eq("ball_snap"), play_tracking["frameId"]
    )
    if snap_frame is None:
        return None

    te_id = candidate.nflId
    rusher_id = candidate.pff_nflIdBlockedPlayer
    te_frames = play_tracking.loc[play_tracking["nflId"].eq(te_id)]
    qb_frames = play_tracking.loc[play_tracking["nflId"].eq(qb_id)]
    if te_frames.empty or qb_frames.empty or pd.isna(rusher_id):
        return None

    release_frame = sustained_release_frame(te_frames, snap_frame)
    qb_by_frame = qb_frames.set_index("frameId")[["x", "y", "team"]]
    opponent_rows = play_tracking.loc[
        play_tracking["team"].ne(qb_by_frame.iloc[0]["team"])
        & play_tracking["team"].ne("football")
        & play_tracking["frameId"].ge(snap_frame)
    ].copy()
    threat_rows = opponent_rows.loc[opponent_rows["nflId"].isin(pass_rusher_ids)].copy()
    if threat_rows.empty:
        return None

    threat_rows = threat_rows.join(qb_by_frame, on="frameId", rsuffix="_qb", how="inner")
    threat_rows["distance_to_qb"] = (
        (threat_rows["x"] - threat_rows["x_qb"]) ** 2
        + (threat_rows["y"] - threat_rows["y_qb"]) ** 2
    ) ** 0.5
    nearest_by_frame = threat_rows.groupby("frameId")["distance_to_qb"].min()
    closing = nearest_by_frame.diff().lt(0)
    threat_frame = first_frame_after(
        nearest_by_frame.le(THREAT_RADIUS_YARDS)
        & closing
        & nearest_by_frame.index.to_series().ge(snap_frame + MIN_THREAT_DELAY_FRAMES),
        nearest_by_frame.index.to_series(),
    )

    pass_frame = first_frame_after(
        play_tracking["event"].eq("pass_forward"), play_tracking["frameId"]
    )
    last_frame = int(play_tracking["frameId"].max())
    evaluation_frame = min(
        frame for frame in (threat_frame, pass_frame, last_frame) if frame is not None
    )

    separation = None
    availability_score = 0.0
    if release_frame is not None and release_frame <= evaluation_frame:
        te_at_evaluation = te_frames.loc[te_frames["frameId"].le(evaluation_frame)].tail(1)
        defenders_at_evaluation = opponent_rows.loc[
            opponent_rows["frameId"].eq(evaluation_frame)
        ]
        if not te_at_evaluation.empty and not defenders_at_evaluation.empty:
            te_row = te_at_evaluation.iloc[0]
            separation = float(
                (
                    (defenders_at_evaluation["x"] - te_row["x"]) ** 2
                    + (defenders_at_evaluation["y"] - te_row["y"]) ** 2
                )
                .pow(0.5)
                .min()
            )
            availability_score = min(separation / OPEN_SEPARATION_YARDS, 1.0)

    threat_or_end_frame = threat_frame if threat_frame is not None else last_frame
    protection_window_seconds = (threat_or_end_frame - snap_frame) / FRAMES_PER_SECOND
    release_time_seconds = (
        None if release_frame is None else (release_frame - snap_frame) / FRAMES_PER_SECOND
    )
    dual_threat_window_seconds = (
        0.0
        if release_time_seconds is None
        else max(protection_window_seconds - release_time_seconds, 0.0) * availability_score
    )

    return {
        "gameId": candidate.gameId,
        "playId": candidate.playId,
        "nflId": te_id,
        "displayName": candidate.displayName,
        "blockedRusherNflId": rusher_id,
        "releaseTimeSeconds": release_time_seconds,
        "protectionWindowSeconds": protection_window_seconds,
        "evaluationFrame": evaluation_frame,
        "nearestDefenderSeparationYards": separation,
        "availabilityScore": availability_score,
        "dualThreatWindowSeconds": dual_threat_window_seconds,
        "threatObserved": threat_frame is not None,
    }


def main() -> None:
    args = parse_args()
    chip_routes, passers, pass_rushers = read_candidates(args.data_dir)
    if args.game_id:
        chip_routes = chip_routes.loc[chip_routes["gameId"].isin(args.game_id)]

    results: list[dict[str, object]] = []
    for game_id, game_candidates in chip_routes.groupby("gameId"):
        tracking_path = args.data_dir / "tracking" / f"tracking_{game_id}.csv"
        if not tracking_path.exists():
            raise FileNotFoundError(f"Missing tracking data: {tracking_path}")

        tracked_plays = set(game_candidates["playId"])
        tracking = pd.read_csv(tracking_path)
        tracking = tracking.loc[tracking["playId"].isin(tracked_plays)]
        game_passers = passers.loc[passers["gameId"].eq(game_id)]
        game_pass_rushers = pass_rushers.loc[pass_rushers["gameId"].eq(game_id)]

        for _, candidate in game_candidates.iterrows():
            qb = game_passers.loc[game_passers["playId"].eq(candidate.playId), "qbNflId"]
            if qb.empty:
                continue
            rusher_ids = set(
                game_pass_rushers.loc[
                    game_pass_rushers["playId"].eq(candidate.playId), "nflId"
                ].astype(int)
            )
            play_tracking = tracking.loc[tracking["playId"].eq(candidate.playId)]
            result = analyze_play(play_tracking, candidate, int(qb.iloc[0]), rusher_ids)
            if result is not None:
                results.append(result)

    output = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} TE chip-and-release rows to {args.output}")


if __name__ == "__main__":
    main()
