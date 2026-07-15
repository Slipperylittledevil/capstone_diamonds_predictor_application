"""
config.py — single source of truth for paths, column groups, and grade orderings.

[Every other module imports from here. Change a path or a grade scale once, and the
 whole project follows. This is the difference between a project and a pile of scripts.]
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# [PATHS] — resolved relative to the repo root, so notebooks and scripts agree
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

RAW_CSV     = DATA_DIR / "diamonds.csv"
CLEAN_CSV   = DATA_DIR / "diamonds_clean.csv"

for d in (DATA_DIR, MODEL_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# [GRADE SCALES] — these are ORDINAL. Worst grade first, best grade last.
#
# This is the single most important modelling decision in the project.
# Unlike 'region' or 'sex' in the insurance data, these categories have a REAL
# ranking. A GIA grader would tell you Ideal > Premium > Very Good > Good > Fair.
# So mapping them to 0,1,2,3,4 is not inventing an order — it is RECORDING one.
#
# One-hot encoding these would THROW AWAY that ordering and force the model to
# rediscover it from scratch. Here, ordinal encoding is the correct choice.
# ---------------------------------------------------------------------------
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]          # 0..4
COLOR_ORDER   = ["J", "I", "H", "G", "F", "E", "D"]                        # 0..6  (D = colourless = best)
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]   # 0..7  (IF = flawless = best)

CUT_MAP     = {v: i for i, v in enumerate(CUT_ORDER)}
COLOR_MAP   = {v: i for i, v in enumerate(COLOR_ORDER)}
CLARITY_MAP = {v: i for i, v in enumerate(CLARITY_ORDER)}

CATEGORICAL = ["cut", "color", "clarity"]
NUMERIC     = ["carat", "depth", "table", "x", "y", "z"]

# [Targets]
CLF_TARGET = "clarity"   # 8-class classification
REG_TARGET = "price"     # regression

N_CLARITY_CLASSES = len(CLARITY_ORDER)

# ---------------------------------------------------------------------------
# [BRAND PALETTE] — used across every plot so the report looks like one thing
# ---------------------------------------------------------------------------
PALETTE = {
    "primary":   "#7C4DFF",   # purple
    "secondary": "#FF80AB",   # blush pink
    "accent":    "#B388FF",   # lavender
    "ink":       "#1A1033",   # deep navy
    "muted":     "#9E9E9E",
}
SEQ_CMAP = "cool"   # [purple -> pink; matches the palette above]
