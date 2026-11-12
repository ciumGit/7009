"""Shared data preparation for the stroke XAI assignment.

Deliberate differences from the repository's utils.DataLoader (all documented
in section 2.1 of the report):

1. The repository calls preprocess_data() on the full dataframe before
   train/validation/test splitting; imputation is therefore fitted on data that
   later becomes the test set. Here every learned transformation lives inside a
   scikit-learn Pipeline and is fitted on the training split only.
2. The repository fills missing BMI with 0 kg/m2. No patient in the file has
   BMI = 0, so this injects an impossible value; the median of the training
   split is used instead.
3. The repository uses train_test_split without stratify, which is unstable for
   a 4.9% positive class. A stratified 60/20/20 split is used here and the test
   split is touched only once, after all choices are frozen on validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "stroke"
ID_COL = "id"

NUMERIC = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # of the remaining 80%


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # The brief spells this value "Govt_jov"; the file uses "Govt_job".
    df["work_type"] = df["work_type"].replace({"Govt_jov": "Govt_job"})
    return df


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def pretty_name(raw: str) -> str:
    """Turn 'cat__work_type_Private' into 'work_type = Private'."""
    raw = raw.split("__", 1)[-1]
    for col in CATEGORICAL:
        if raw.startswith(col + "_"):
            return f"{col} = {raw[len(col) + 1:]}"
    return raw


def group_of(raw: str) -> str:
    """Parent variable of a transformed column, used to aggregate one-hot blocks."""
    raw = raw.split("__", 1)[-1]
    for col in CATEGORICAL:
        if raw.startswith(col + "_"):
            return col
    return raw


def prepare(csv_path: str | Path, scale_numeric: bool = True, seed: int = SEED):
    """Return X/y frames plus the fitted preprocessor and split indices."""
    df = load_raw(csv_path)
    X = df.drop(columns=[ID_COL, TARGET])
    y = df[TARGET].astype(int)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=seed
    )
    rel_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1 - rel_val, stratify=y_tmp, random_state=seed
    )

    pre = build_preprocessor(scale_numeric=scale_numeric)
    Z_tr = pre.fit_transform(X_tr)
    Z_val = pre.transform(X_val)
    Z_te = pre.transform(X_te)

    names = [pretty_name(n) for n in pre.get_feature_names_out()]
    raw_names = list(pre.get_feature_names_out())
    groups = [group_of(n) for n in raw_names]

    to_frame = lambda A, idx: pd.DataFrame(A, columns=names, index=idx)

    return dict(
        raw=df,
        X_train=to_frame(Z_tr, X_tr.index),
        X_val=to_frame(Z_val, X_val.index),
        X_test=to_frame(Z_te, X_te.index),
        y_train=y_tr, y_val=y_val, y_test=y_te,
        preprocessor=pre,
        feature_names=names,
        groups=groups,
        source_rows=dict(train=X_tr.index, val=X_val.index, test=X_te.index),
    )


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
