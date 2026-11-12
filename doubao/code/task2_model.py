"""Task 2: train the classifier, evaluate it on a held-out test set, build a
SHAP TreeExplainer and verify that base value + sum(SHAP) reconstructs the
explained probability.

Run from the repository root:  python analysis/task2_model.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 2021
OUT, FIG, MOD = "results", "figures", "models"
for d in (OUT, FIG, MOD):
    os.makedirs(d, exist_ok=True)

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]

num_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

# 60 / 20 / 20 stratified split; preprocessing and resampling see training rows only
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                            random_state=SEED)

pre.fit(X_tr)
feature_names = [c.replace("num__", "").replace("cat__", "") for c in pre.get_feature_names_out()]
Z_tr = pd.DataFrame(pre.transform(X_tr), columns=feature_names)
Z_val = pd.DataFrame(pre.transform(X_val), columns=feature_names)
Z_te = pd.DataFrame(pre.transform(X_te), columns=feature_names)

# Oversample only the training rows
ros = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
Z_tr_bal, y_tr_bal = ros.fit_resample(Z_tr, y_tr)

# Modest grid search on the validation set, selection by positive-class F1 at 0.5
grid = [{"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20}]
best = None
for params in grid:
    m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    m.fit(Z_tr_bal, y_tr_bal)
    p = m.predict_proba(Z_val)[:, 1]
    f1 = f1_score(y_val, (p >= 0.5).astype(int))
    if best is None or f1 > best[0]:
        best = (f1, params, m)
val_f1, best_params, rf = best

# Threshold chosen on validation by maximising positive-class F1
p_val = rf.predict_proba(Z_val)[:, 1]
ths = np.unique(np.round(p_val, 4))
scores = [(f1_score(y_val, (p_val >= t).astype(int)), t) for t in ths]
thr = float(max(scores)[1])

# Test evaluation
p_te = rf.predict_proba(Z_te)[:, 1]
pred = (p_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
metrics = {
    "seed": SEED,
    "split": {"train": int(len(y_tr)), "validation": int(len(y_val)), "test": int(len(y_te)),
              "train_positive": int(y_tr.sum()), "validation_positive": int(y_val.sum()),
              "test_positive": int(y_te.sum())},
    "oversampled_train": int(len(y_tr_bal)),
    "best_params": best_params,
    "validation_f1_at_0.5": round(float(val_f1), 4),
    "threshold": round(thr, 4),
    "n_features": len(feature_names),
    "feature_names": feature_names,
    "test": {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "average_precision": round(float(average_precision_score(y_te, p_te)), 4),
        "brier": round(float(brier_score_loss(y_te, p_te)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "prevalence": round(float(y_te.mean()), 4),
    },
}

# Subgroup recall where the test fold has enough positive cases
band = pd.cut(X_te["age"], bins=[0, 18, 35, 50, 65, 85], labels=["<18", "18-34", "35-49", "50-64", "65+"])
subgroup = {}
band_reset = band.reset_index(drop=True)
for b in band_reset.cat.categories:
    idx = band_reset[band_reset == b].index
    yy = y_te.reset_index(drop=True).loc[idx]
    pp = pred[idx]
    subgroup[str(b)] = {"n": int(len(idx)), "positives": int(yy.sum()),
                        "recall": round(float(recall_score(yy, pp, zero_division=0)), 4)}
metrics["subgroup_recall_age"] = subgroup

# Calibration check (reliability of the probability output)
frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")
metrics["calibration"] = {"mean_pred": [round(float(v), 3) for v in mean_pred],
                          "frac_pos": [round(float(v), 3) for v in frac_pos]}

# ---------------- SHAP: TreeExplainer, probability of class 1 ----------------
bg = shap.sample(Z_tr, 50, random_state=SEED)
explainer = shap.TreeExplainer(rf, data=bg, model_output="probability",
                               feature_perturbation="interventional")
expl_idx = Z_te.sample(n=min(300, len(Z_te)), random_state=SEED).index
Z_ex = Z_te.loc[expl_idx].reset_index(drop=True)
sv = explainer.shap_values(Z_ex, check_additivity=False)
if isinstance(sv, list):
    sv1 = sv[1]
elif np.ndim(sv) == 3:
    sv1 = sv[:, :, 1]
else:
    sv1 = sv
base = float(np.ravel(explainer.expected_value)[1])

recon = base + sv1.sum(axis=1)
model_out = rf.predict_proba(Z_ex)[:, 1]
resid = np.abs(recon - model_out)
metrics["explainer"] = {
    "class": "shap.TreeExplainer",
    "model_output": "probability",
    "feature_perturbation": "interventional",
    "background_rows": 50,
    "background_source": "training rows after preprocessing, shap.sample seed 2021",
    "explained_rows": int(len(Z_ex)),
    "explained_source": "held-out test rows, sample seed 2021",
    "base_value": round(base, 5),
    "max_abs_residual": round(float(resid.max()), 6),
    "mean_abs_residual": round(float(resid.mean()), 6),
    "positive_class_index": 1,
    "shap_version": shap.__version__,
}

# Store artefacts for later tasks
Z_ex.to_csv(f"{OUT}/explained_features.csv", index=False)
pd.DataFrame(sv1, columns=feature_names).to_csv(f"{OUT}/shap_values_class1.csv", index=False)
X_te.reset_index(drop=True).loc[expl_idx].assign(
    y_true=y_te.reset_index(drop=True).loc[expl_idx].values,
    p_stroke=p_te[expl_idx]).to_csv(f"{OUT}/explained_raw_rows.csv", index=False)
meta = {"base_value": base, "threshold": thr, "feature_names": feature_names}
json.dump(meta, open(f"{OUT}/explainer_meta.json", "w"), indent=2)
json.dump(metrics, open(f"{OUT}/metrics_task2.json", "w"), indent=2)
joblib.dump({"model": rf, "preprocessor": pre, "threshold": thr, "params": best_params},
            f"{MOD}/task2_pipeline.joblib")

# Keep the full test frame with predictions for case selection
te_frame = X_te.reset_index(drop=True).copy()
te_frame["y_true"] = y_te.reset_index(drop=True)
te_frame["p_stroke"] = p_te
te_frame["y_pred"] = pred
te_frame.to_csv(f"{OUT}/test_predictions.csv", index=False)

# Mean absolute SHAP ranking
mean_abs = pd.DataFrame({"feature": feature_names,
                         "mean_abs_shap": np.abs(sv1).mean(axis=0)}).sort_values(
    "mean_abs_shap", ascending=False).reset_index(drop=True)
mean_abs.to_csv(f"{OUT}/global_importance.csv", index=False)

print(json.dumps({k: metrics[k] for k in ["split", "best_params", "threshold", "test", "subgroup_recall_age", "explainer"]}, indent=2))
print(mean_abs.head(10).to_string(index=False))
