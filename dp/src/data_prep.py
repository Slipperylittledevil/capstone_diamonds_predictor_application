"""
data_prep.py — load, clean, engineer, encode, split.

[Everything upstream of a model lives here. The models never touch a raw CSV; they
 ask this module for a matrix. That separation is what makes the three models
 comparable — they are all fed from the same pipe.]
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (RAW_CSV, CLEAN_CSV, RANDOM_STATE,
                        CUT_MAP, COLOR_MAP, CLARITY_MAP,
                        CLARITY_ORDER, CUT_ORDER, COLOR_ORDER)


# ===========================================================================
# [1] LOAD
# ===========================================================================
def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the Kaggle diamonds.csv."""
    df = pd.read_csv(path)

    # [Kaggle's export carries a leftover row-number column. It is a serial number
    #  with no meaning, but a distance-based or gradient-based model will happily
    #  learn from it. Drop it.]
    for junk in ["Unnamed: 0", "index"]:
        if junk in df.columns:
            df = df.drop(columns=junk)
    return df


# ===========================================================================
# [2] CLEAN
# ===========================================================================
def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove impossible rows, duplicates, and physically absurd outliers."""
    n0 = len(df)
    report = {}

    # [Missing values: the diamonds dataset famously has NONE. We check anyway,
    #  because "I checked and there were none" is a finding; "I assumed" is not.]
    report["missing_values"] = int(df.isna().sum().sum())
    df = df.dropna()

    # [Duplicates: ~146 exact duplicate rows exist. They are almost certainly
    #  data-entry artefacts, and they let the same row land in BOTH train and test,
    #  which quietly inflates every score you report.]
    dupes = int(df.duplicated().sum())
    report["duplicates_removed"] = dupes
    df = df.drop_duplicates()

    # [Zero dimensions: ~20 rows have x, y, or z equal to 0. A diamond with zero
    #  width does not exist. This is a missing value wearing a numeric disguise —
    #  and it would never be caught by isna(). Always sanity-check numeric ranges
    #  against physical reality.]
    zero_dims = int(((df[["x", "y", "z"]] == 0).any(axis=1)).sum())
    report["zero_dimension_rows"] = zero_dims
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

    # [Absurd outliers: a couple of rows list y = 58.9 mm or z = 31.8 mm — that is a
    #  diamond the size of a fist, on a row priced like a small one. Data-entry error.
    #  A single row like this can drag a regression line noticeably.]
    before = len(df)
    df = df[(df["y"] < 20) & (df["z"] < 20)]
    df = df[(df["depth"].between(45, 75)) & (df["table"].between(40, 90))]
    report["absurd_outliers_removed"] = before - len(df)

    report["rows_before"] = n0
    report["rows_after"] = len(df)

    if verbose:
        print("--- CLEANING REPORT ---")
        for k, v in report.items():
            print(f"  {k:<26} {v}")
        print(f"  {'retained':<26} {len(df)/n0:.2%}")

    return df.reset_index(drop=True)


# ===========================================================================
# [3] FEATURE ENGINEERING
# ===========================================================================
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features that encode gemmological knowledge the raw columns don't."""
    df = df.copy()

    # [volume: x*y*z. Carat is a WEIGHT; volume is a SIZE. They're related but not
    #  identical — a poorly-cut stone is bulky for its weight. Giving the model both
    #  lets it learn the difference.]
    df["volume"] = df["x"] * df["y"] * df["z"]

    # [density proxy: weight / volume. A high value means the stone is dense for its
    #  size, which correlates with a deep, weight-retaining cut.]
    df["density"] = df["carat"] / df["volume"].replace(0, np.nan)

    # [symmetry: a round brilliant should have x ~= y. Deviation from 1.0 is a
    #  measurable cut defect that no single raw column captures.]
    df["xy_ratio"] = df["x"] / df["y"]

    # [log_carat: carat is heavily right-skewed and its effect on price is
    #  multiplicative, not additive. The log makes the relationship closer to linear.]
    df["log_carat"] = np.log1p(df["carat"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


# ===========================================================================
# [4] ENCODE — ordinal, because these grades are genuinely ranked
# ===========================================================================
def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map cut/color/clarity text grades to their true ordinal ranks."""
    df = df.copy()
    df["cut_rank"]     = df["cut"].map(CUT_MAP)
    df["color_rank"]   = df["color"].map(COLOR_MAP)
    df["clarity_rank"] = df["clarity"].map(CLARITY_MAP)

    if df[["cut_rank", "color_rank", "clarity_rank"]].isna().any().any():
        bad = df[df[["cut_rank", "color_rank", "clarity_rank"]].isna().any(axis=1)]
        raise ValueError(f"Unrecognised grade values in {len(bad)} rows — check config maps.")

    return df


def build(verbose: bool = True) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer -> encode. Caches to disk."""
    df = load_raw()
    df = clean(df, verbose=verbose)
    df = engineer(df)
    df = encode(df)
    df.to_csv(CLEAN_CSV, index=False)
    if verbose:
        print(f"\nSaved -> {CLEAN_CSV}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    return df


# ===========================================================================
# [5] FEATURE MATRICES — one per task, and they are NOT the same
# ===========================================================================

# [CLASSIFICATION: predict clarity.
#  We may use price, because in the business scenario the price is known — you have a
#  stone in front of you with a listed price and you want to sanity-check its clarity
#  grade. We must NOT use clarity_rank (that IS the answer).]
CLF_FEATURES = ["carat", "log_carat", "depth", "table", "price",
                "x", "y", "z", "volume", "density", "xy_ratio",
                "cut_rank", "color_rank"]

# [REGRESSION: predict price.
#  We may use clarity_rank (a graded stone's clarity is known before it is priced).
#  We must NOT use price, log_price, or price_per_carat. That last one is the classic
#  trap: price_per_carat * carat = price exactly. A model given it would score R^2 =
#  1.00 and be completely worthless, because at prediction time you don't have it.
#  If a feature is a rearrangement of the target, it is LEAKAGE, not a feature.]
REG_FEATURES = ["carat", "log_carat", "depth", "table",
                "x", "y", "z", "volume", "density", "xy_ratio",
                "cut_rank", "color_rank", "clarity_rank"]

# [CLUSTERING: segment the inventory.
#  No target at all. We pick the axes a buyer actually shops along: size, price,
#  and the three quality grades.]
CLU_FEATURES = ["carat", "price", "cut_rank", "color_rank", "clarity_rank", "volume"]


def split_scaled(df, features, target, test_size=0.2, val_size=0.1,
                 stratify=False, random_state=RANDOM_STATE):
    """
    Split into train / validation / test, then scale using TRAIN statistics only.

    [The scaler is fit on X_train and merely APPLIED to X_val and X_test. If you fit it
     on everything, the test set's own mean and standard deviation leak into training,
     and your reported score is dishonest. The scaler is part of the model.]

    [Three-way split, not two: the VALIDATION set is what EarlyStopping watches while
     training. If early stopping watched the test set, you'd be tuning against it, and
     it would stop being a held-out set.]
    """
    X = df[features].values.astype("float32")
    y = df[target].values

    strat = y if stratify else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=strat)

    strat2 = y_tmp if stratify else None
    rel = test_size / (test_size + val_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel, random_state=random_state, stratify=strat2)

    scaler = StandardScaler().fit(X_tr)          # [fit on TRAIN only]
    X_tr  = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    return X_tr, X_val, X_te, y_tr, y_val, y_te, scaler
