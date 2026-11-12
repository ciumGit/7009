"""Task 6: a second, structurally different black box model (a neural network)
explained with Kernel SHAP.

The split, predictors and resampling rule are identical to Task 2 so that the
comparison is like for like.

Run from the repository root:  python analysis/task6_alt.py
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 2021
OUT, FIG, MOD = "results", "figures", "models"

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]

num_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

# identical split (same seed and sequence as Task 2)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                            random_state=SEED)
pre.fit(X_tr)
fnames = [c.replace("num__", "").replace("cat__", "") for c in pre.get_feature_names_out()]
Z_tr = pd.DataFrame(pre.transform(X_tr), columns=fnames)
Z_val = pd.DataFrame(pre.transform(X_val), columns=fnames)
Z_te = pd.DataFrame(pre.transform(X_te), columns=fnames)

ros = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
Z_trb, y_trb = ros.fit_resample(Z_tr, y_tr)

svc = SVC(C=0.5, kernel="rbf", gamma="scale", cache_size=500)
mlp = CalibratedClassifierCV(svc, method="sigmoid", cv=3)
t0 = time.time()
mlp.fit(Z_trb, y_trb)
train_seconds = round(time.time() - t0, 1)

# threshold: same rule as Task 2, chosen on validation by positive-class F1
p_val = mlp.predict_proba(Z_val)[:, 1]
ths = np.unique(np.round(p_val, 4))
thr = float(max((f1_score(y_val, (p_val >= t), ), t) for t in ths)[1])

p_te = mlp.predict_proba(Z_te)[:, 1]
pred = (p_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
metrics = {
    "model": "RBF SVM (C=0.5, gamma=scale) with Platt scaling, 3-fold calibration",
    "explainer": "shap.KernelExplainer on predict_proba, class 1",
    "training_seconds": train_seconds,
    "threshold": round(thr, 4),
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
    },
}

# ---------------- Kernel SHAP ----------------
bg = shap.kmeans(Z_tr, 20)
explainer = shap.KernelExplainer(lambda z: mlp.predict_proba(np.asarray(z))[:, 1], bg)
eval_idx = Z_te.sample(n=60, random_state=SEED).index
Z_ev = Z_te.loc[eval_idx].reset_index(drop=True)
t1 = time.time()
sv_global = explainer.shap_values(Z_ev, nsamples=300, l1_reg="num_features(10)")
kernel_seconds = round(time.time() - t1, 1)
base_k = float(np.ravel(explainer.expected_value)[0])

recon = base_k + sv_global.sum(axis=1)
out = mlp.predict_proba(Z_ev)[:, 1]
resid = np.abs(recon - out)

# Same patient as Task 3, located in the identical test fold
case = json.load(open(f"{OUT}/case_task3.json"))
rec = case["raw_record"]
match = X_te.reset_index(drop=True)
mask = np.ones(len(match), dtype=bool)
for k, v in rec.items():
    mask &= (match[k].astype(str) == str(v)).values
case_row = int(np.where(mask)[0][0])
z_case = pd.DataFrame(pre.transform(X_te.iloc[[case_row]]), columns=fnames)
t2 = time.time()
sv_case = explainer.shap_values(z_case, nsamples=800, l1_reg="num_features(10)")
kernel_local_seconds = round(time.time() - t2, 1)

# ---------------- Figures ----------------
plain = {"age": "Age", "bmi": "BMI", "avg_glucose_level": "Blood sugar",
         "hypertension": "High blood pressure", "heart_disease": "Heart disease",
         "gender": "Gender", "ever_married": "Marital status", "work_type": "Job type",
         "Residence_type": "Residence", "smoking_status": "Smoking"}


def group_of(name):
    for g in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        if name.startswith(g + "_"):
            return g
    return name


def grouped_rows(sv):
    G = pd.DataFrame(0.0, index=range(len(sv)), columns=list(plain))
    for j, f in enumerate(fnames):
        G[group_of(f)] += sv[:, j]
    return G


def display_value(g):
    if g == "age":
        return f"{rec['age']:.0f}y"
    if g == "bmi":
        return f"{rec['bmi']:.1f}"
    if g == "avg_glucose_level":
        return f"{rec['avg_glucose_level']:.0f}"
    if g == "hypertension":
        return "yes" if int(rec["hypertension"]) == 1 else "no"
    if g == "heart_disease":
        return "yes" if int(rec["heart_disease"]) == 1 else "no"
    return str(rec[g]).replace("_", " ")


Gc = grouped_rows(sv_case.reshape(1, -1)).iloc[0]
top = Gc.abs().sort_values(ascending=False).head(6).index

# Local explanation for the alternative model: grouped contribution chart
FX = base_k + float(Gc.sum())
items = [(g, float(v)) for g, v in Gc.items() if abs(float(v)) >= 0.01]
items.sort(key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(9.8, 4.2))
labels = [f"{plain[g]} = {display_value(g)}" for g, _ in items]
vals = [v for _, v in items]
colors = ["#C0504D" if v > 0 else "#4C78A8" for v in vals]
ax.barh(labels, vals, color=colors, height=0.55)
ax.axvline(0, color="black", lw=0.8)
lim = max(abs(min(vals)), abs(max(vals))) * 1.25
ax.set_xlim(-lim, lim)
for y, v in enumerate(vals):
    ax.text(v + (0.006 if v > 0 else -0.006), y, f"{v:+.3f}",
            ha="left" if v > 0 else "right", va="center", fontsize=8.5)
ax.set_xlabel("Change in predicted stroke probability (SVM + Kernel SHAP)")
ax.set_title(f"Local explanation of the alternative model, f(x) = {FX:.2f}, base = {base_k:.2f}",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_kernel_local.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Global explanation: grouped importance bar for the alternative model
Gg = grouped_rows(sv_global)
gimp = pd.DataFrame({"group": Gg.columns, "mean_abs": Gg.abs().mean().values})
gimp["pretty"] = gimp["group"].map(plain)
gimp = gimp.sort_values("mean_abs", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.5, 4.3))
head = gimp.head(8)[::-1]
ax.barh(head["pretty"], head["mean_abs"], color="#7F9E7F", height=0.6)
ax.set_xlabel("Mean |SHAP| (grouped, probability scale)")
ax.set_title("Global importance, SVM with Kernel SHAP (60 held-out patients)", fontsize=10)
for y, v in enumerate(head["mean_abs"]):
    ax.text(v + head["mean_abs"].max() * 0.01, y, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0, head["mean_abs"].max() * 1.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_kernel_global.png", dpi=200, bbox_inches="tight")
plt.close(fig)

metrics["explainer_detail"] = {
    "background": "shap.kmeans of 5110-row training fold, 20 clusters",
    "nsamples_global": 300,
    "nsamples_local": 800,
    "l1_reg": "num_features(10)",
    "explained_rows_global": 60,
    "base_value": round(base_k, 5),
    "max_abs_residual": round(float(resid.max()), 6),
    "mean_abs_residual": round(float(resid.mean()), 6),
    "kernel_seconds_global": kernel_seconds,
    "kernel_seconds_local": kernel_local_seconds,
    "case_row_in_test": case_row,
    "case_fx": round(float(FX), 4),
    "case_base": round(base_k, 5),
    "case_contributions": {plain[g]: round(float(Gc[g]), 4) for g in top},
    "top5_global": gimp.head(5)[["pretty", "mean_abs"]].to_dict("records"),
}
json.dump(metrics, open(f"{OUT}/metrics_task6.json", "w"), indent=2, default=float)
Gg.to_csv(f"{OUT}/task6_grouped_shap.csv", index=False)
pd.DataFrame(sv_global, columns=fnames).to_csv(f"{OUT}/task6_shap_values.csv", index=False)
joblib.dump({"model": mlp, "preprocessor": pre, "threshold": thr}, f"{MOD}/task6_svm.joblib")

print(json.dumps(metrics, indent=2, default=float))
