"""Search, rank, filter, and compare TE Versatility Scout player profiles.

Run ``.venv/bin/python -m src.coach_report`` first, then for example:

    .venv/bin/python -m src.player_search --search kelce
    .venv/bin/python -m src.player_search --rank-by chip-to-route --top 10
    .venv/bin/python -m src.player_search --similar-to "Travis Kelce" --similarity-mode combined
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.coach_report import AXES, ROLE_SHARE_COLUMNS

INDEX_PATH = Path("output/coach_report/search_index.csv")
RANK_COLUMNS = {
    "overall": "versatility_profile_score",
    "protection": "Protection Impact",
    "chip-to-route": "Chip-to-Route Value",
    "route-threat": "Route Threat",
    "coverage-stress": "Coverage Stress",
    "open-field": "Open-Field Creation",
    "deployment": "deployment_breadth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search TE Versatility Scout player profiles.")
    parser.add_argument("--search", help="Case-insensitive player-name search.")
    parser.add_argument("--team", help="Three-letter team abbreviation.")
    parser.add_argument(
        "--archetype",
        help="Case-insensitive match against performance or deployment archetype.",
    )
    parser.add_argument("--min-snaps", type=int, default=0, help="Minimum tracked TE snaps.")
    parser.add_argument(
        "--rank-by",
        choices=sorted(RANK_COLUMNS),
        default="overall",
        help="Profile dimension used for rankings.",
    )
    parser.add_argument("--top", type=int, default=10, help="Maximum rows to print.")
    parser.add_argument("--similar-to", help="Player name or nflId for nearest-player results.")
    parser.add_argument(
        "--similarity-mode",
        choices=["performance", "role", "combined"],
        default="combined",
        help="Use performance axes, deployment roles, or a 70/30 combined score.",
    )
    parser.add_argument("--include-ineligible", action="store_true", help="Include players below 100 snaps.")
    parser.add_argument("--output", type=Path, help="Optional CSV destination for the result.")
    return parser.parse_args()


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} does not exist. Run `.venv/bin/python -m src.coach_report` first."
        )
    return pd.read_csv(INDEX_PATH)


def filter_index(index: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = index.copy()
    if not args.include_ineligible:
        filtered = filtered.loc[filtered.profile_eligible]
    if args.search:
        filtered = filtered.loc[filtered.displayName.str.contains(args.search, case=False, na=False)]
    if args.team:
        filtered = filtered.loc[filtered.team.eq(args.team.upper())]
    if args.archetype:
        term = args.archetype.casefold()
        performance = filtered.performance_archetype.fillna("").str.casefold().str.contains(term)
        deployment = filtered.deployment_archetype.fillna("").str.casefold().str.contains(term)
        filtered = filtered.loc[performance | deployment]
    return filtered.loc[filtered.snaps.ge(args.min_snaps)].copy()


def find_target(index: pd.DataFrame, query: str) -> pd.Series:
    exact_id = index.loc[index.nflId.astype(str).eq(query)]
    exact_name = index.loc[index.displayName.str.casefold().eq(query.casefold())]
    matches = pd.concat([exact_id, exact_name]).drop_duplicates("nflId")
    if len(matches) != 1:
        names = ", ".join(matches.displayName) if len(matches) else "none"
        raise ValueError(f"Expected one player for {query!r}; matches: {names}")
    return matches.iloc[0]


def similarity(index: pd.DataFrame, target: pd.Series, mode: str) -> pd.DataFrame:
    pool = index.loc[index.nflId.ne(target.nflId)].copy()
    performance = pool[list(AXES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    target_performance = pd.to_numeric(target[list(AXES)], errors="coerce").to_numpy(dtype=float)
    roles = pool[ROLE_SHARE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    target_roles = pd.to_numeric(target[ROLE_SHARE_COLUMNS], errors="coerce").to_numpy(dtype=float)
    performance_distance = np.sqrt(np.nanmean((performance - target_performance) ** 2, axis=1))
    role_distance = np.sqrt(np.nanmean((roles - target_roles) ** 2, axis=1)) * 100
    pool["performance_similarity"] = (100 - performance_distance).clip(0, 100)
    pool["role_similarity"] = (100 - role_distance).clip(0, 100)
    if mode == "performance":
        pool["similarity_score"] = pool.performance_similarity
    elif mode == "role":
        pool["similarity_score"] = pool.role_similarity
    else:
        pool["similarity_score"] = 0.70 * pool.performance_similarity + 0.30 * pool.role_similarity
    return pool.sort_values("similarity_score", ascending=False)


def main() -> None:
    args = parse_args()
    index = load_index()
    if args.similar_to:
        target = find_target(index, args.similar_to)
        result = similarity(filter_index(index, args), target, args.similarity_mode)
        result.insert(0, "target_player", target.displayName)
    else:
        result = filter_index(index, args).sort_values(RANK_COLUMNS[args.rank_by], ascending=False)

    columns = [
        column
        for column in [
            "target_player",
            "similarity_score",
            "performance_similarity",
            "role_similarity",
            "profile_rank",
            "displayName",
            "team",
            "snaps",
            "performance_archetype",
            "deployment_archetype",
            "versatility_profile_score",
            *AXES,
            "deployment_breadth",
        ]
        if column in result
    ]
    result = result.head(args.top)[columns]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, float_format="%.2f")
        print(f"Wrote {len(result)} rows to {args.output}")
    else:
        print(result.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
