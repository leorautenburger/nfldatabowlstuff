# TE Versatility Scout: Implementation Handoff

## Product goal

Build a tight-end decision-support tool for an NFL coach, scout, or
broadcaster. The tool should answer:

> Which TEs fit a particular offensive need: protect the QB, help protection
> and still become an outlet, win routes, stress coverage, create after the
> catch, or contribute to the run game?

The submission requires Python code, outputs, and a short README. It will be
judged on football insight, technical execution, communication, and originality
across triage, rubric grading, and deliberation rounds.

The user-facing product is a coach-facing player profile, not a single opaque
ranking. Component metrics and play-level evidence must remain inspectable.

## Repository and branch state

- The complete metric suite and coach-profile integration are on
  **`te-metrics-pipeline`**.
- `main` contains a newer uploaded `te_run_metrics.csv`, but that upload removed
  the broader pipeline code and output artifacts from `main`.
- Do not overwrite or discard the pipeline branch. Reconcile branches only
  after the run-data integration is validated.
- The coach-profile generator is `src/coach_report.py`.
- Its current output is `output/coach_report/players.csv` and
  `output/coach_report/summary.json`.
- Run it with:

  ```bash
  .venv/bin/python -m src.coach_report
  ```

- Spider charts are intentionally opt-in:

  ```bash
  .venv/bin/python -m src.coach_report --render-spider-charts
  ```

  Do not generate or commit spider-chart artifacts until the profile
  composition and labels have been reviewed.

## Data and reproducibility

The available Big Data Bowl data covers 2021 Weeks 1-8 and is pass-play
focused. The processed local data and raw tracking files are not necessarily
committed to GitHub. Metric outputs are committed and can be used to build the
coach profile.

The project uses a shared environment and processed data convention:

- `src/common/config.py` defines paths, shared thresholds, and the Week 1-6
  train / Week 7-8 test split.
- `src/common/io.py` loads processed Parquet data.
- `output/<metric>/players.csv`, `plays.csv`, and `summary.json` are the
  standard metric outputs.

Do not treat raw, untracked local directories as submission artifacts.

## Metric inventory

### Keep in the final product

| Metric | Primary football role | Current output |
| --- | --- | --- |
| Protection Value Added (PVA) | Pure pass-protection value from a pressure counterfactual. | `output/protection_value_added/` |
| Edge Seal Sustainability | Sustained protection against the assigned edge threat. | `output/edge_seal_sustainability/` |
| Protection-to-Availability Value (PAV) | Chip-and-release value: extra protection time while the TE remains available. | `output/protection_to_availability_value/` |
| Chip-to-Separation Return (CSR) | Separation retained after a chip or delayed release. | `output/chip_to_separation_return/` |
| Route Release Tax | Cost of delayed release; lower is better. | `output/route_release_tax/` |
| Eligible Threat Rate | Timely availability as a receiving option. | `output/eligible_threat_rate/` |
| Matchup Separation Over Expected | Route-winning separation after context adjustment. | `output/matchup_soe/` |
| Middle-of-Field Access | Availability in middle-field throwing windows. | `output/middle_of_field_access/` |
| Coverage Gravity | Additional defensive attention attracted by the TE. | `output/coverage_gravity/` |
| Constraint Value | Change in defensive alignment caused by TE alignment. | `output/constraint_value/` |
| Red-Zone Conflict | TE’s ability to create route/run/protection conflicts near the goal line. | `output/red_zone_conflict/` |
| YAC Runway | Post-catch open-field opportunity. | `output/yac_runway/` |
| Dual-Threat Deployment Index (DDI) | Context-adjusted breadth of TE role usage. | `output/deployment_index/` |

### Do not use as final profile metrics

| Artifact | Decision |
| --- | --- |
| `pav_prototype.py` | Development-only descriptive precursor. It has no counterfactual estimate of added protection time. |
| `te_protection_value_added.csv` | Legacy top-level PVA output; superseded by `output/protection_value_added/`. |
| `te_expected_yards_metrics.csv` | Supporting production/context table only. Do not add it to the spider profile because it overlaps with route-threat and YAC components. |

## PAV decision

There are two similarly named final-stage concepts and one prototype:

1. **Prototype dual-threat window:** post-release time before a threat, weighted
   by separation. Do not surface it.
2. **PVA:** expected pressure difference with TE protection removed. This is a
   pure protection metric and applies to blocks and chips.
3. **PAV:** matched chip-and-release protection-window lift multiplied by
   receiving availability. This is the primary novel metric.

PAV definition:

```text
PAV =
  (observed protection window
   - expected no-chip protection window from matched standard TE routes)
  * receiving availability score
```

The current PAV implementation uses:

- PFF `CH` / `SR` chip-and-release labels,
- a 2-yard closing QB threat zone,
- separation, passing lane, viable depth, and sideline room for availability,
- cross-fitted, eight-neighbor matching against standard TE routes,
- Weeks 1-6 training and Weeks 7-8 held-out testing.

PAV is still observational, so it needs opportunity counts and caveats. It
should be the project’s headline measure, not hidden inside generic protection.

## Target player-profile composition

The final tool should show **six performance axes**. Scores should be
league-relative 0-100 percentiles, sign-aligned so higher is better, and use
empirical-Bayes shrunk component values where available.

### 1. Protection Impact

```text
0.60 * PVA + 0.40 * Edge Seal Sustainability
```

This measures pure pocket-protection value. Do not include PAV here; that
would double-count the chip-and-release role.

### 2. Chip-to-Route Value

```text
0.40 * PAV
+ 0.40 * Chip-to-Separation Return
+ 0.20 * inverted Route Release Tax
```

This is the signature dual-threat axis and should be prominent in the
submission narrative.

### 3. Route Threat

```text
0.34 * Eligible Threat Rate
+ 0.33 * Matchup Separation Over Expected
+ 0.33 * Middle-of-Field Access
```

### 4. Coverage Stress

```text
0.50 * Coverage Gravity
+ 0.25 * Constraint Value
+ 0.25 * Red-Zone Conflict
```

### 5. Open-Field Creation

```text
1.00 * YAC Runway
```

For this metric, use `yacoe_shrunk`, not `value_shrunk`: the latter is constant
in the current output and produces a flat profile axis.

### 6. Run-Game Impact (future integration)

The new `te_run_metrics.csv` on `main` provides:

- run/pass snap rates and run tilt,
- rush EPA and success with the TE on/off field,
- edge-run EPA and success,
- short-yardage conversions,
- heavy-package usage,
- average defensive box count,
- TE carries and rushing yards.

The proposed score is:

```text
0.40 * rush EPA on/off
+ 0.25 * rush success on/off
+ 0.20 * edge-run EPA
+ 0.15 * short-yardage conversion
```

Use `run_tilt`, heavy-package share, and box count as deployment context, not
as performance credit. Establish minimum opportunity rules for offensive runs
and short-yardage plays before ranking players.

### Current provisional integration

The pipeline now includes `src.metrics.run_game_impact`, which creates:

- `output/run_game_impact/players.csv`;
- `output/run_game_impact/matches.csv`;
- `output/run_game_impact/summary.json`.

It matches all 84 source rows by normalized player name and team, records the
match method for every row, applies the minimums above, renormalizes weights
when a component lacks enough opportunities, and shrinks scores toward the
league midpoint. It is included as the sixth app axis but visibly labeled
**provisional**.

The integration is not fully validated until all of the following are complete:

1. Document its source and construction methodology.
2. Validate the season/week scope against the Big Data Bowl sample.
3. Create and validate a `gsis_id` to `nflId` crosswalk; do not join solely on
   player name.
4. Define sample thresholds and uncertainty handling.
5. Produce standard `output/run_game_impact/players.csv`, `plays.csv` or
   supporting rows, and `summary.json` artifacts.

## Deployment Index: required use

DDI is a **role-breadth and context metric**, not a performance metric. It
uses normalized Shannon entropy across these five context-adjusted roles:

1. Inline route
2. Detached route
3. Chip-and-release
4. Standard pass block
5. Stay-in/non-conventional pass block

The implementation inverse-propensity reweights player snaps to a league
formation/personnel context mix, caps extreme weights, and applies
empirical-Bayes shrinkage. This avoids mistaking scheme-driven usage for true
role versatility.

Use DDI in the final tool as:

- a displayed **Deployment Breadth** value and percentile;
- five adjusted role-share bars;
- an archetype input and comparison filter;
- a way to explain *how* a player achieved their performance profile.

Do not place DDI in the six-axis performance average. A TE should not receive
extra performance credit merely for being asked to perform more roles.

Suggested archetypes:

- **Chip-and-Release Specialist:** high adjusted chip-and-release share;
- **Inline Protector:** high adjusted block share;
- **Receiving Specialist:** high adjusted route share;
- **Balanced TE:** no dominant role and high DDI;
- **Limited Sample:** insufficient total or role-specific opportunities.

## Coach-facing output

The final player table should include:

- player, team, total snaps, route/chip/block/run opportunities;
- six axis scores and their component counts;
- Deployment Breadth/Index and five role shares;
- archetype;
- optional overall profile score;
- links or keys to underlying play-level evidence.

The overall score should be secondary. Until Run-Game Impact has complete
source coverage and documented provenance, use an unweighted mean of the five
validated pass-game axes. Show the provisional sixth run axis separately;
never include DDI in the performance score.

### Discovery and comparison requirements

The final tool must support:

- name and team search;
- ranking by any performance axis or component metric;
- filters for minimum snaps, performance archetype, deployment archetype, and
  role usage;
- similar-player search in three modes:
  - **performance:** distance over performance axes;
  - **role:** distance over DDI-adjusted role shares;
  - **combined:** 70% performance similarity and 30% role similarity.

Current implementation:

```bash
.venv/bin/python -m src.player_search --search kelce
.venv/bin/python -m src.player_search --rank-by chip-to-route --top 10
.venv/bin/python -m src.player_search --similar-to "Travis Kelce" --similarity-mode combined
```

`output/coach_report/search_index.csv` is the machine-readable index for a
future UI. It currently contains the five validated pass-game axes, deployment
breadth, adjusted role shares, and both archetype types. Extend it with the
run-game axis only after the run-data requirements above are met.

Spider charts can be valuable in a presentation, but only after metric
composition is frozen. They should show player percentile versus a dashed
league-median baseline and always display opportunity counts.

## Required validation and communication

Before final submission:

1. Run every metric and the integrated profile command end-to-end.
2. Verify all input metric tables are present and every output has expected
   row counts and required columns.
3. Check score direction and column selection, especially:
   - invert Route Release Tax;
   - use `yacoe_shrunk` for YAC Runway;
   - do not double-count PAV in Protection Impact;
   - use shrunk values when valid.
4. Verify the run-data ID crosswalk and provenance.
5. Keep the README short: football question, one command, output files,
   profile-axis meanings, and caveat.
6. Keep metric `summary.json` files and play-level tables as evidence for
   judges who want technical detail.

## Submission narrative

The central story is:

> TE Versatility Scout separates what a tight end is **asked to do**
> (deployment breadth, role shares, and run usage) from how well they
> **perform** in those roles (protection, chip-and-route value, route threat,
> coverage stress, open-field creation, and run-game impact).

PAV provides the novel hook: identify tight ends who buy their quarterback
time **without removing themselves as receiving options**.
