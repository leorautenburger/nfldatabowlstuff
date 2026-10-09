"""Build a provisional run-game impact metric from the uploaded run table.

The source table's upstream construction is not documented in this repository.
This module therefore publishes its match audit and labels all outputs
provisional. It must not be presented as tracking-derived causal evidence.

Run from the repository root:
    .venv/bin/python -m src.metrics.run_game_impact
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("output/run_game_impact")
SOURCE = Path("te_run_metrics.csv")
MIN_OFF_RUNS = 30
MIN_EDGE_RUNS = 20
MIN_SHORT_YARDAGE = 5
SHRINKAGE_RUNS = 50


def normalize(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def percentile(values: pd.Series, eligible: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index)
    result.loc[eligible] = values.loc[eligible].rank(pct=True, method="average") * 100
    return result


def match_players(run: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    players = players[["nflId", "displayName", "team"]].copy()
    players["name_key"] = players.displayName.map(normalize)
    players["team_key"] = players.team.map(normalize)
    run = run.copy()
    run["name_key"] = run.name.map(normalize)
    run["team_keys"] = run.teams.fillna("").str.split("/").apply(
        lambda teams: {normalize(team) for team in teams if team}
    )

    rows = []
    for source in run.itertuples(index=False):
        candidates = players.loc[players.name_key.eq(source.name_key)]
        exact = candidates.loc[candidates.team_key.isin(source.team_keys)]
        if len(exact) == 1:
            matched, method = exact.iloc[0], "exact_name_team"
        elif len(candidates) == 1:
            matched, method = candidates.iloc[0], "unique_name_multi_team_source"
        else:
            matched, method = None, "unmatched_or_ambiguous"
        row = source._asdict()
        row.update(
            {
                "nflId": None if matched is None else int(matched.nflId),
                "displayName": None if matched is None else matched.displayName,
                "team": None if matched is None else matched.team,
                "match_method": method,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source table: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    run = pd.read_csv(SOURCE)
    profile_players = pd.read_csv("output/te_metrics_final.csv")
    matched = match_players(run, profile_players)
    matched.to_csv(OUT / "matches.csv", index=False)

    scored = matched.loc[matched.nflId.notna()].copy()
    scored["nflId"] = scored.nflId.astype(int)
    scored["eligible_base"] = scored.n_off_runs.ge(MIN_OFF_RUNS)
    scored["epa_score"] = percentile(scored.rush_epa_onoff, scored.eligible_base)
    scored["success_score"] = percentile(scored.rush_success_onoff_raw, scored.eligible_base)
    scored["edge_score"] = percentile(
        scored.edge_rush_epa_on, scored.eligible_base & scored.n_edge_runs.ge(MIN_EDGE_RUNS)
    )
    scored["short_score"] = percentile(
        scored.short_yardage_conv,
        scored.eligible_base & scored.n_short_yardage.ge(MIN_SHORT_YARDAGE),
    )
    component_columns = ["epa_score", "success_score", "edge_score", "short_score"]
    weights = np.array([0.40, 0.25, 0.20, 0.15])
    values = scored[component_columns].to_numpy(dtype=float)
    available = ~np.isnan(values)
    numerator = np.nansum(values * weights, axis=1)
    denominator = (available * weights).sum(axis=1)
    scored["value"] = np.divide(
        numerator, denominator, out=np.full(len(scored), np.nan), where=denominator > 0
    )
    shrink_weight = scored.n_off_runs / (scored.n_off_runs + SHRINKAGE_RUNS)
    scored["value_shrunk"] = 50 + shrink_weight * (scored.value - 50)
    scored["n"] = scored.run_snaps
    scored["higher_is_better"] = True
    scored["provisional"] = True
    scored["component_count"] = available.sum(axis=1)
    scored["meets_min_runs"] = scored.eligible_base
    players = scored[
        [
            "nflId",
            "displayName",
            "team",
            "n",
            "value",
            "value_shrunk",
            "higher_is_better",
            "provisional",
            "component_count",
            "meets_min_runs",
            "run_snaps",
            "n_off_runs",
            "n_edge_runs",
            "n_short_yardage",
            "rush_epa_onoff",
            "rush_success_onoff_raw",
            "edge_rush_epa_on",
            "short_yardage_conv",
            "run_tilt",
            "heavy_share",
            "avg_box_on",
            "epa_score",
            "success_score",
            "edge_score",
            "short_score",
            "match_method",
        ]
    ].sort_values("value_shrunk", ascending=False)
    players.to_csv(OUT / "players.csv", index=False, float_format="%.4f")
    summary = {
        "metric": "Run-Game Impact",
        "status": "provisional",
        "source": {
            "file": str(SOURCE),
            "provenance": "Undocumented uploaded table from the repository main branch.",
            "warning": (
                "The upstream methodology is not available in this repository. "
                "This score is a transparent integration layer, not validated tracking evidence."
            ),
        },
        "player_matching": {
            "method": "exact normalized player name + team, with unique-name fallback for multi-team source rows",
            "source_rows": int(len(run)),
            "matched_rows": int(scored.nflId.nunique()),
            "unmatched_rows": int(matched.nflId.isna().sum()),
            "match_methods": matched.match_method.value_counts().to_dict(),
            "audit_file": "matches.csv",
        },
        "score": {
            "formula": (
                "0.40 * rush EPA on/off percentile + 0.25 * rush success on/off percentile + "
                "0.20 * edge-run EPA percentile + 0.15 * short-yardage conversion percentile; "
                "available weights are renormalized."
            ),
            "minimums": {
                "offensive_runs": MIN_OFF_RUNS,
                "edge_runs": MIN_EDGE_RUNS,
                "short_yardage_plays": MIN_SHORT_YARDAGE,
            },
            "shrinkage": f"score shrunk toward 50 with n_off_runs / (n_off_runs + {SHRINKAGE_RUNS})",
        },
        "usage_context_not_scored": ["run_tilt", "heavy_share", "avg_box_on"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(players)} provisional run-impact player rows to {OUT}")


if __name__ == "__main__":
    main()
