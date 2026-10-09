# TE Versatility Scout: Measuring Dual-Threat Tight End Value

TE Versatility Scout is a coach-facing NFL Big Data Bowl tool for comparing
tight ends as protectors, chip-and-release threats, route winners, coverage
stressors, and open-field creators.

## Football question

**Which tight end best fits a specific offensive need—and can they help protect
the quarterback without removing themselves as a receiving option?**

The signature metric is **Protection-to-Availability Value (PAV)**: matched
extra protection-window time from a chip-and-release assignment, weighted by
the TE's receiving availability.

## View the interactive visualization

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.serve_visualization
```

Open [http://127.0.0.1:8000/web/](http://127.0.0.1:8000/web/). The application
supports player/team search, rankings, archetype filters, deployment filters,
performance/role similarity, player profiles, and an in-app methodology page.

## Included materials

| Path | Contents |
| --- | --- |
| `src/` | Python metric, profile, search, and visualization-server code. |
| `web/` | Dependency-free interactive visualization. |
| `output/` | Player tables, play-level evidence, validation outputs, and metric definitions. |
| `te_run_metrics.csv` | Uploaded run-game source table used for the provisional run axis. |

The profile summarizes Protection Impact, Chip-to-Route Value, Route Threat,
Coverage Stress, Open-Field Creation, and provisional Run-Game Impact.
Deployment Breadth explains role usage and archetypes but does not improve a
player's performance score.

## Rebuild the final profile

```bash
.venv/bin/python -m src.metrics.run_game_impact
.venv/bin/python -m src.coach_report
```

The UI reads `output/coach_report/search_index.json`. Every component metric
also retains `players.csv`, play-level evidence, and `summary.json` under
`output/<metric>/`.

## Scope and limitations

The tracking study covers 2021 Weeks 1–8 and is pass-play focused. Metrics are
transparent observational estimates, not proprietary NFL grades or causal
claims.

Run-Game Impact is **provisional**: the uploaded run table has no upstream
methodology in this repository. It is visible and searchable, but excluded
from the overall score and performance-similarity calculation. Players without
a matching run row show `N/A`, not a zero score.

See the in-app **Methodology** page for every formula, statistic, weighting
rule, and caveat.
