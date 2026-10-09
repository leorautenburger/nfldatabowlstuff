# Tight End Dual-Threat Metrics

## Objective

Evaluate tight ends as both receiving threats and pass-protection contributors.
The central question is:

> What is the offensive value of a tight end's decision to block, chip, or release into a route?

This proposal uses the NFL Big Data Bowl regional-event dataset's tracking,
play-context, and PFF scouting data. The metrics below are custom-derived;
they are not provided as columns in the source data.

## Available inputs

| Source | Useful fields |
| --- | --- |
| Tracking | `x`, `y`, `s`, `a`, `dis`, `o`, `dir`, `event`, `frameId`, and `time` |
| PFF scouting | `pff_role`, `pff_positionLinedUp`, `pff_nflIdBlockedPlayer`, `pff_blockType`, `pff_backFieldBlock`, and pressure outcomes |
| Play data | formation, personnel, coverage, play action, dropback type, down, distance, and play result |
| Player data | official position and player identifiers |

Use the ball-snap event as time zero. Restrict the primary analysis to players
whose official position is `TE`, then use `pff_role` to identify whether their
assignment on each play was a route, pass block, or chip-and-release action.

## Core metrics

### Route Release Tax

**Definition:** Time from the ball snap to the first frame in which the tight
end begins their route.

**Interpretation:** A larger value represents a delayed release, often caused
by a chip, stay-in protection responsibility, or play-action action. Report it
with route depth and target/catch outcomes; delayed releases are not
automatically negative.

### Chip-to-Separation Return

**Definition:** Separation gained after a chip, normalized by release delay and
route depth.

One transparent starting formulation is:

```text
CSR = separation at the catch/throw frame
      - expected separation for route depth and release time
```

**Interpretation:** Identifies tight ends that help in protection and still
become timely, viable receiving options.

### Eligible Threat Rate

**Definition:** The share of dropbacks in which the tight end:

1. runs a route,
2. releases before a defined time threshold, and
3. reaches a viable receiving position before the pass is thrown.

Define a viable receiving position from separation, field location, route
direction, and the time-to-throw distribution. Publish the threshold and run a
sensitivity check rather than treating it as a hidden rule.

### Coverage Gravity

**Definition:** Additional defensive attention toward the tight end relative
to comparable aligned receivers and formations.

Candidate inputs include nearest-defender distance, number of defenders within
a defined radius, bracket-like positioning, and changes in safety depth.

**Interpretation:** Captures a tight end creating space for other receivers
without receiving a target. The red zone is a particularly relevant split.

### Matchup Separation Over Expected

**Definition:** Actual tight-end separation at route landmarks minus expected
separation given alignment, coverage, defender position, route depth, and time
since snap.

```text
SOE = observed separation - E(separation | context)
```

**Interpretation:** Makes receiving comparisons fairer across difficult
coverage assignments and route usage.

### Middle-of-Field Access

**Definition:** Probability that the tight end offers a viable throwing window
over the middle at fixed dropback times, such as 1.5, 2.5, and 3.5 seconds.

**Interpretation:** Quantifies the reliable check-down and intermediate-window
role frequently expected from tight ends.

### YAC Runway

**Definition:** Space ahead of and around the tight end at the catch point,
combined with speed and acceleration into the catch.

**Interpretation:** Separates yards-after-catch opportunity created by scheme
from opportunity created by the player.

### Protection Value Added

**Definition:** Change in expected pressure probability or expected
time-to-threat when the tight end stays in to block or chips before releasing.

```text
PVA = expected pressure without TE assistance
      - expected pressure with observed TE assistance
```

Estimate expected pressure using protection count, rusher alignment,
pass-rusher identity or quality, QB movement, dropback type, play action, and
time since snap. Evaluate the metric against PFF hurry, hit, and sack labels.

**Interpretation:** Measures the protection benefit of the tight end while
accounting for the offensive line, quarterback movement, and opponent.

### Protection-to-Availability Value

**Definition:** The additional time a tight end creates for the quarterback
while still becoming a viable receiving option.

```text
PAV =
  (expected time to QB threat without TE assistance
   - observed time to QB threat with TE assistance)
  * receiving availability score
```

The full metric is counterfactual: it needs an expected-pressure model to
estimate the no-assistance outcome. The repository's `pav_prototype.py`
implements a transparent descriptive precursor instead:

```text
dualThreatWindowSeconds =
  max(protectionWindowSeconds - releaseTimeSeconds, 0)
  * availabilityScore
```

It measures the time remaining after the TE's release before a threat reaches
the QB, weighted by the TE's separation at the pass, threat, or play-end
frame. It should not be described as causal added time.

### Edge Seal Sustainability

**Definition:** During a TE pass-block assignment, the time the assigned rusher
remains outside a defined QB threat zone or behind the QB-depth plane.

**Interpretation:** A tracking-based measure of block quality that is more
granular than allowed-pressure counts.

### Dual-Threat Deployment Index

**Definition:** Formation-adjusted usage mix across inline routes, detached
routes, chip-and-release assignments, stay-in blocks, and standard pass-block
snaps.

**Interpretation:** Identifies versatile tight ends and prevents comparisons
between receiving specialists and protection specialists.

### Constraint Value

**Definition:** Difference in defensive alignment, box count, safety depth, or
coverage tendencies when the tight end is inline versus detached or absent.

**Interpretation:** Estimates how tight-end alignment changes defensive
behavior before the snap.

### Red-Zone Conflict Score

**Definition:** In red-zone and goal-to-go situations, quantify whether a
defense must account for the tight end as a route threat, run-fit player, and
protector simultaneously.

**Interpretation:** Highlights tight-end versatility where space is limited and
alignment ambiguity is especially valuable.

## Recommended implementation order

1. **Route Release Tax:** directly measurable from snap timing and trajectory.
2. **Dual-Threat Deployment Index:** descriptive and based mainly on PFF role
   and block-type labels.
3. **Chip-to-Separation Return:** combines the first two ideas with nearest
   defender distance.
4. **Protection Value Added:** the primary novel analysis; use a controlled
   expected-pressure model and validate it against PFF pressure outcomes.

## Reporting principles

- Show components alongside any composite score.
- Control for formation, coverage, route depth, down/distance, play action,
  QB movement, and opponent.
- Separate inline, detached, and backfield alignments.
- Report uncertainty and sample size, especially for chips, protection snaps,
  and red-zone splits.
- Treat expected-value models as transparent, reproducible estimates rather
  than proprietary equivalents of NFL tracking metrics.

## Optional composite

Use a composite only as a secondary summary:

```text
TE Dual-Threat Value =
  0.35 * Route Value
  + 0.25 * Protection Value
  + 0.20 * Chip-to-Separation Return
  + 0.10 * Coverage Gravity
  + 0.10 * Deployment Versatility
```

The component metrics remain primary because a receiving specialist, an inline
blocker, and a chip-and-release specialist can each create offensive value in
different ways.
