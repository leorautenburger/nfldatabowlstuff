"""Build coach-facing TE versatility profiles and dependency-free spider charts.

Run from the repository root:
    .venv/bin/python -m src.coach_report

Add --render-spider-charts to create SVG visualizations after reviewing the
profile table.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("output/coach_report")
MIN_PROFILE_SNAPS = 100


@dataclass(frozen=True)
class Metric:
    slug: str
    axis: str
    weight: float
    higher_is_better: bool = True
    value_column: str | None = None


METRICS = (
    Metric("protection_to_availability_value", "Protection Impact", 0.40),
    Metric("protection_value_added", "Protection Impact", 0.40),
    Metric("edge_seal_sustainability", "Protection Impact", 0.20),
    Metric("chip_to_separation_return", "Chip-to-Route Value", 0.60),
    Metric("route_release_tax", "Chip-to-Route Value", 0.40, higher_is_better=False),
    Metric("eligible_threat_rate", "Route Threat", 0.34),
    Metric("matchup_soe", "Route Threat", 0.33),
    Metric("middle_of_field_access", "Route Threat", 0.33),
    Metric("coverage_gravity", "Coverage Stress", 0.50),
    Metric("constraint_value", "Coverage Stress", 0.25),
    Metric("red_zone_conflict", "Coverage Stress", 0.25),
    Metric("yac_runway", "Open-Field Creation", 1.00, value_column="yacoe_shrunk"),
    Metric("run_game_impact", "Run-Game Impact", 1.00),
)
AXES = tuple(dict.fromkeys(metric.axis for metric in METRICS))
VALIDATED_AXES = tuple(axis for axis in AXES if axis != "Run-Game Impact")
ROLE_SHARE_COLUMNS = [
    "wshare_shrunk_inline_route",
    "wshare_shrunk_detached_route",
    "wshare_shrunk_chip_release",
    "wshare_shrunk_stay_in_block",
    "wshare_shrunk_standard_pass_block",
]


def metric_score(metric: Metric) -> pd.DataFrame:
    path = Path("output") / metric.slug / "players.csv"
    frame = pd.read_csv(path)
    value_column = metric.value_column or (
        "value_shrunk" if "value_shrunk" in frame.columns else "value"
    )
    required = ["nflId", "n", value_column]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    score = frame[required].copy()
    score = score.rename(columns={"n": f"{metric.slug}_n", value_column: f"{metric.slug}_value"})
    valid = score[f"{metric.slug}_n"].gt(0) & score[f"{metric.slug}_value"].notna()
    percentile = score.loc[valid, f"{metric.slug}_value"].rank(pct=True, method="average") * 100
    if not metric.higher_is_better:
        percentile = 100 - percentile
    score[f"{metric.slug}_percentile"] = np.nan
    score.loc[valid, f"{metric.slug}_percentile"] = percentile
    return score


def build_profiles() -> tuple[pd.DataFrame, dict[str, object]]:
    base = pd.read_csv("output/te_metrics_final.csv")
    profiles = base[["nflId", "displayName", "team", "snaps", "route_snaps", "chip_snaps", "block_snaps"]].copy()
    for metric in METRICS:
        profiles = profiles.merge(metric_score(metric), on="nflId", how="left")
    deployment = pd.read_csv("output/deployment_index/players.csv")
    deployment = deployment[
        ["nflId", "value_shrunk", *ROLE_SHARE_COLUMNS, "archetype", "archetype_reliable"]
    ].rename(
        columns={
            "value_shrunk": "deployment_breadth",
            "archetype": "deployment_archetype",
        }
    )
    profiles = profiles.merge(deployment, on="nflId", how="left")

    for axis in AXES:
        axis_metrics = [metric for metric in METRICS if metric.axis == axis]
        score_columns = [f"{metric.slug}_percentile" for metric in axis_metrics]
        weights = np.array([metric.weight for metric in axis_metrics])
        values = profiles[score_columns].to_numpy(dtype=float)
        available = ~np.isnan(values)
        weighted_sum = np.nansum(values * weights, axis=1)
        available_weight = (available * weights).sum(axis=1)
        axis_score = np.full(len(profiles), np.nan)
        np.divide(weighted_sum, available_weight, out=axis_score, where=available_weight > 0)
        profiles[axis] = axis_score
        profiles[f"{axis}_components"] = available.sum(axis=1)

    profiles["profile_eligible"] = profiles.snaps.ge(MIN_PROFILE_SNAPS)
    profiles["run_game_available"] = profiles["Run-Game Impact"].notna()
    profiles["performance_archetype"] = profiles.apply(assign_archetype, axis=1)
    profiles["versatility_profile_score"] = profiles[list(VALIDATED_AXES)].mean(axis=1)
    profiles["profile_rank"] = (
        profiles.loc[profiles.profile_eligible, "versatility_profile_score"]
        .rank(method="min", ascending=False)
    )
    profiles = profiles.sort_values(
        ["profile_eligible", "versatility_profile_score"], ascending=[False, False]
    ).reset_index(drop=True)
    metadata = {
        "axes": list(AXES),
        "metric_mapping": {
            axis: [
                {
                    "metric": metric.slug,
                    "weight": metric.weight,
                    "higher_is_better": metric.higher_is_better,
                    "value_column": metric.value_column or "value_shrunk/value",
                }
                for metric in METRICS
                if metric.axis == axis
            ]
            for axis in AXES
        },
        "min_profile_snaps": MIN_PROFILE_SNAPS,
        "percentile_definition": (
            "Percentile among all TEs with a non-missing component score. "
            "Where supplied, empirical-Bayes shrunk values are used."
        ),
        "deployment_context": {
            "definition": (
                "Deployment Breadth is the context-adjusted Dual-Threat Deployment Index. "
                "It is shown for archetyping and comparison but excluded from the performance score."
            ),
            "role_share_columns": ROLE_SHARE_COLUMNS,
        },
        "provisional_axes": ["Run-Game Impact"],
        "overall_score_axes": list(VALIDATED_AXES),
    }
    return profiles, metadata


def assign_archetype(player: pd.Series) -> str:
    protection = player["Protection Impact"]
    chip_route = player["Chip-to-Route Value"]
    route = player["Route Threat"]
    coverage = player["Coverage Stress"]
    yac = player["Open-Field Creation"]
    if pd.isna([protection, chip_route, route, coverage, yac]).any():
        return "Limited sample"
    if protection >= 70 and route >= 70:
        return "Dual-Threat Creator"
    if protection >= 75:
        return "Inline Protector"
    if route >= 75 and coverage >= 65:
        return "Receiving Mismatch"
    if coverage >= 75:
        return "Coverage Stressor"
    if yac >= 75:
        return "Open-Field Creator"
    return "Balanced TE"


def point(value: float, angle: float, center: float, radius: float) -> tuple[float, float]:
    scaled = value / 100 * radius
    return center + scaled * math.cos(angle), center + scaled * math.sin(angle)


def svg_for_player(player: pd.Series) -> str:
    size, center, radius = 480, 240, 145
    axes = [axis for axis in AXES if pd.notna(player[axis])]
    angles = [(-math.pi / 2) + index * 2 * math.pi / len(axes) for index in range(len(axes))]
    values = [float(player[axis]) for axis in axes]

    grid = []
    for level in (25, 50, 75, 100):
        vertices = [point(level, angle, center, radius) for angle in angles]
        grid.append(
            '<polygon points="{}" fill="none" stroke="#cbd5e1" stroke-width="1"/>'.format(
                " ".join(f"{x:.1f},{y:.1f}" for x, y in vertices)
            )
        )
    spokes = []
    labels = []
    for axis, angle in zip(axes, angles):
        x, y = point(100, angle, center, radius)
        lx, ly = point(122, angle, center, radius)
        anchor = "middle" if abs(lx - center) < 15 else ("start" if lx > center else "end")
        spokes.append(f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="axis">{html.escape(axis)}</text>'
        )
    baseline = [point(50, angle, center, radius) for angle in angles]
    profile = [point(value, angle, center, radius) for value, angle in zip(values, angles)]
    title = f"{player.displayName} ({player.team})"
    subtitle = (
        f"{player.performance_archetype} | {int(player.snaps)} snaps | "
        f"Profile score: {player.versatility_profile_score:.0f}"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
<style>
  .title {{ font: 700 20px sans-serif; fill: #0f172a; }}
  .subtitle {{ font: 13px sans-serif; fill: #475569; }}
  .axis {{ font: 12px sans-serif; fill: #334155; }}
  .key {{ font: 11px sans-serif; fill: #475569; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{center}" y="28" text-anchor="middle" class="title">{html.escape(title)}</text>
<text x="{center}" y="48" text-anchor="middle" class="subtitle">{html.escape(subtitle)}</text>
{''.join(grid)}{''.join(spokes)}
<polygon points="{' '.join(f'{x:.1f},{y:.1f}' for x, y in baseline)}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5 4"/>
<polygon points="{' '.join(f'{x:.1f},{y:.1f}' for x, y in profile)}" fill="#2563eb" fill-opacity="0.28" stroke="#1d4ed8" stroke-width="3"/>
<text x="25" y="452" class="key">Blue: player percentile | Dashed: league median (50th percentile)</text>
</svg>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build coach-facing TE versatility profiles.")
    parser.add_argument(
        "--render-spider-charts",
        action="store_true",
        help="Render one dependency-free SVG spider chart for every eligible TE.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    profiles, metadata = build_profiles()
    output_columns = [
        "profile_rank",
        "nflId",
        "displayName",
        "team",
        "snaps",
        "route_snaps",
        "chip_snaps",
        "block_snaps",
        "profile_eligible",
        "run_game_available",
        "performance_archetype",
        "deployment_archetype",
        "archetype_reliable",
        "deployment_breadth",
        *ROLE_SHARE_COLUMNS,
        "versatility_profile_score",
        *AXES,
        *[f"{axis}_components" for axis in AXES],
    ]
    profiles[output_columns].to_csv(OUT / "players.csv", index=False, float_format="%.2f")
    profiles[output_columns].to_csv(OUT / "search_index.csv", index=False, float_format="%.2f")
    profiles[output_columns].to_json(OUT / "search_index.json", orient="records", indent=2)
    if args.render_spider_charts:
        chart_dir = OUT / "spider_charts"
        chart_dir.mkdir(exist_ok=True)
        for _, row in profiles.loc[profiles.profile_eligible].iterrows():
            filename = f"{int(row.nflId)}_{row.displayName.lower().replace(' ', '_').replace('.', '')}.svg"
            (chart_dir / filename).write_text(svg_for_player(row))

    metadata["eligible_profiles"] = int(profiles.profile_eligible.sum())
    metadata["total_tes"] = int(len(profiles))
    metadata["spider_charts_rendered"] = args.render_spider_charts
    metadata["output_files"] = {
        "players": "players.csv",
        "search_index": "search_index.csv",
        "search_index_json": "search_index.json",
        "spider_charts": "spider_charts/<nflId>_<player>.svg",
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    chart_status = (
        f"and {metadata['eligible_profiles']} spider charts"
        if args.render_spider_charts
        else "without spider charts"
    )
    print(f"Wrote {len(profiles)} TE profiles {chart_status} to {OUT}")


if __name__ == "__main__":
    main()
