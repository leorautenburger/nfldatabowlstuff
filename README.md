# TE Versatility Scout

An NFL Big Data Bowl tool for comparing tight ends as blockers, receivers, and
dual-threat players. It turns tracking and PFF scouting data into transparent,
play-level evidence and a coach-facing player profile.

## Football question

**Which TEs create the most offensive value for a specific role: protect the
QB, become a viable outlet after helping protection, win routes, stress
coverage, or create after the catch?**

The metric suite includes Protection-to-Availability Value (PAV), Protection
Value Added, Chip-to-Separation Return, route availability, separation over
expected, coverage gravity, red-zone conflict, and YAC runway.

## Run

After producing the component metric outputs in `output/<metric>/`, build the
coach-facing table:

```bash
.venv/bin/python -m src.coach_report
```

This writes:

- `output/coach_report/players.csv` — ranked player profiles, five 0-100
  percentile axes, opportunity counts, and archetypes.
- `output/coach_report/summary.json` — metric-to-axis mapping and thresholds.

Spider charts are intentionally opt-in while profiles are reviewed:

```bash
.venv/bin/python -m src.coach_report --render-spider-charts
```

## Discover players

The profile output supports player lookup, category rankings, team/archetype
filters, and nearest-player comparisons:

```bash
.venv/bin/python -m src.player_search --search kelce
.venv/bin/python -m src.player_search --rank-by chip-to-route --top 10
.venv/bin/python -m src.player_search --archetype "receiving specialist"
.venv/bin/python -m src.player_search --similar-to "Travis Kelce" --similarity-mode combined
```

Similarity modes are `performance` (the profile axes), `role` (context-adjusted
deployment shares), and `combined` (70% performance, 30% role).

## Interactive visualization

Run the profile generator, then serve the local web app:

```bash
.venv/bin/python -m src.coach_report
.venv/bin/python -m src.serve_visualization
```

Open [http://127.0.0.1:8000/web/](http://127.0.0.1:8000/web/) to search,
filter, rank, compare similar players, and inspect a live performance radar.
Use `Ctrl+C` in the serving terminal to stop it.

## Coach profile axes

| Axis | Football interpretation |
| --- | --- |
| Protection Impact | Pocket protection, edge sealing, and chip-and-release PAV. |
| Chip-to-Route Value | Protection help without losing route availability. |
| Route Threat | Separation, timely eligibility, and middle-of-field access. |
| Coverage Stress | Defensive attention, alignment constraint, and red-zone conflict. |
| Open-Field Creation | Expected post-catch runway and YAC opportunity. |

Scores are league-relative percentiles. Where available, empirical-Bayes
shrunk estimates are used; all component metrics, sample sizes, and play-level
evidence remain available in their own `output/<metric>/` directories.

Deployment Breadth and its role shares are shown as context and used for
archetypes/similarity, but are excluded from the performance score.

## Caveat

These are transparent, tracking-derived observational metrics—not proprietary
NFL grades or causal claims. Use player profiles with their opportunity counts
and the underlying play rows, especially for low-volume roles such as
chip-and-release.
