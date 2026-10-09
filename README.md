# TE Versatility Scout

An NFL Big Data Bowl decision-support tool for comparing tight ends as
protectors, receivers, chip-and-release threats, coverage stressors, and
open-field creators.

**Football question:** Which TE best fits a specific offensive need—and can
they help protect the QB without removing themselves as a receiving option?

## View the interactive tool

The committed outputs are ready to explore:

```bash
.venv/bin/python -m src.serve_visualization
```

Open [http://127.0.0.1:8000/web/](http://127.0.0.1:8000/web/) to search,
filter, rank, compare similar players, and inspect player profiles. The
in-app [Methodology](http://127.0.0.1:8000/web/methodology.html) page defines
every displayed statistic and formula.

To rebuild the search index from component outputs:

```bash
.venv/bin/python -m src.metrics.run_game_impact
.venv/bin/python -m src.coach_report
```

## Profile dimensions

| Axis | What it measures |
| --- | --- |
| Protection Impact | PVA and sustained edge protection. |
| Chip-to-Route Value | PAV, chip-to-separation return, and release cost. |
| Route Threat | Timely availability, separation over expected, and middle-field access. |
| Coverage Stress | Defensive attention, alignment constraint, and red-zone conflict. |
| Open-Field Creation | YAC runway and post-catch opportunity. |
| Run-Game Impact † | Run EPA/success on-off, edge-run EPA, and short-yardage conversion. |

Scores are league-relative percentiles. Player profile scores average the five
fully covered pass-game dimensions; Deployment Breadth informs archetypes and
role similarity but never increases a performance score.

† Run-Game Impact is provisional: `te_run_metrics.csv` has no upstream
methodology in this repository. It is shown and searchable but excluded from
overall scores and performance similarity. Players without a source-table row
see a five-spoke radar and `N/A` rather than a zero run score.

## Outputs and caveats

- `output/coach_report/search_index.csv` is the search/UI data source.
- `output/<metric>/players.csv`, `plays.csv`, and `summary.json` retain player
  samples, play evidence, definitions, validation, and thresholds.
- `output/run_game_impact/matches.csv` audits every run-data player match.

These are transparent observational estimates, not proprietary NFL grades or
causal claims. Interpret rankings with opportunity counts and the underlying
play-level evidence.
