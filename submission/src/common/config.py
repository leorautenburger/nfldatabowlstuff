"""Shared configuration for the TE dual-threat metrics pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "repo" / "data"
PROC = ROOT / "data" / "processed"
TRACK_DIR = PROC / "tracking"          # one parquet per game, standardized coords
OUT = ROOT / "output"

FPS = 10.0
FIELD_W = 53.3

# Hold-out design: split by week (whole games) so no play/drive leaks across sets.
# Weeks 1-6 train (expected-value models fit + cross-fit here), weeks 7-8 test.
TRAIN_WEEKS = [1, 2, 3, 4, 5, 6]
TEST_WEEKS = [7, 8]
N_FOLDS = 5  # GroupKFold by gameId inside train weeks
SEED = 42

# Release detection (shared by Route Release Tax, CSR, Eligible Threat Rate, ...)
RELEASE_MIN_DISP = 1.5   # yards moved from snap position (downfield or lateral)
RELEASE_MIN_SPEED = 2.5  # yards/sec, must hold for RELEASE_SUSTAIN frames
RELEASE_SUSTAIN = 3

# Viable receiving position (shared by Eligible Threat Rate and Middle-of-Field Access)
VIABLE_MIN_SEP = 2.5        # yards to nearest defender
VIABLE_MIN_LANE = 1.0       # yards: min defender distance to QB->receiver segment
VIABLE_DEPTH = (-2.0, 25.0) # yards relative to line of scrimmage
VIABLE_SIDELINE_BUFFER = 1.0

RED_ZONE_YARDS = 20  # LOS within 20 yards of opponent goal
