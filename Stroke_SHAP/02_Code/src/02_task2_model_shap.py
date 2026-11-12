"""Task 2 - stroke classifier, SHAP explainer, and additivity validation.

The 03_shap.py demonstrates SHAP on a random forest.
gradient-boosted tree (XGBoost) is used instead,
for three reasons that are stated in report section 3.1:

  * the raw model output of a boosted tree ensemble is a log-odds, so the SHAP
    additivity identity base + sum(contributions) = model output holds exactly
    and is verifiable to machine precision; a random forest's "raw" margin is
    a sum of per-tree votes, which is harder to interpret;
  * it keeps task 6 clean: task 6 must explain a *different* black-box type,
    and an SVM with KernelSHAP is then unambiguous;
  * with only 21 features and ~3,000 training rows the model trains in seconds,
    leaving the compute budget to KernelSHAP in task 6.

Differences from the 03_shap.py (recorded in report 3.2):
  * the repository evaluates a single hard-coded test row (start_index=1);
    here all 1,022 test rows are explained and metrics cover the full test split;
  * the repository explains shap_values[1] against expected_value[1] without
    stating the output scale; here the scale (log-odds of stroke = 1, linked to
    probability through the logistic function) is stated explicitly and the
    additivity check base + sum(SHAP) = model output is run on every test row;
  * random over-sampling is dropped (it duplicates the 149 positive training
    rows until the classes are balanced, which inflates apparent recall on
    duplicated patients); scale_pos_weight plays the same role without copies.

"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prep import SEED, prepare, save_json  # noqa: E402

DATA = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

t_start = time.time()
d = prepare(DATA)
X_tr, y_tr = d["X_train"], d["y_train"]
X_va, y_va = d["X_val"], d["y_val"]
X_te, y_te = d["X_test"], d["y_test"]
spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
print(f"splits: train {X_tr.shape} ({y_tr.sum()} pos), val {X_va.shape} ({y_va.sum()} pos), "
      f"test {X_te.shape} ({y_te.sum()} pos); scale_pos_weight {spw:.2f}")

# ---------------------------------------------------------------- model tuning
grid = [
    {"max_depth": md, "learning_rate": lr, "min_child_weight": mcw}
    for md in (3, 4, 6)
    for lr in (0.05, 0.1)
    for mcw in (1, 5)
]
rows = []
for g in grid:
    m = xgb.XGBClassifier(
        n_estimators=300, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=SEED, n_jobs=-1, eval_metric="logloss", **g,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_va)[:, 1]
    rows.append((g, roc_auc_score(y_va, p), average_precision_score(y_va, p)))
    print(f"{g}  val AUC {rows[-1][1]:.4f}  val AP {rows[-1][2]:.4f}")

best = max(rows, key=lambda r: r[1] + r[2])[0]
print("selected hyper-parameters:", best)

model = xgb.XGBClassifier(
    n_estimators=300, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
    scale_pos_weight=spw, random_state=SEED, n_jobs=-1, eval_metric="logloss", **best,
)
model.fit(X_tr, y_tr)

# ------------------------------------------------- decision threshold on val
p_val = model.predict_proba(X_va)[:, 1]
thr_grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
f1s = [f1_score(y_va, (p_val >= t).astype(int), zero_division=0) for t in thr_grid]
thr = float(thr_grid[int(np.argmax(f1s))])
print(f"decision threshold {thr:.2f} (val F1 for stroke=1 {max(f1s):.3f})")

# ------------------------------------------------------------- test metrics
p_te = model.predict_proba(X_te)[:, 1]
y_hat = (p_te >= thr).astype(int)
cm = confusion_matrix(y_te, y_hat)
metrics = dict(
    model="XGBClassifier (gradient-boosted trees)",
    params={"n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.8,
            "reg_lambda": 1.0, "scale_pos_weight": round(spw, 2),
            "random_state": SEED, **best},
    threshold=thr,
    threshold_rule="maximise F1 for stroke=1 on the validation split (0.02..0.97 grid)",
    test_n=int(len(y_te)),
    test_pos=int(y_te.sum()),
    accuracy=round(float(accuracy_score(y_te, y_hat)), 4),
    balanced_accuracy=round(float(balanced_accuracy_score(y_te, y_hat)), 4),
    precision_pos=round(float(precision_score(y_te, y_hat, zero_division=0)), 4),
    recall_pos=round(float(recall_score(y_te, y_hat, zero_division=0)), 4),
    f1_pos=round(float(f1_score(y_te, y_hat, zero_division=0)), 4),
    f1_macro=round(float(f1_score(y_te, y_hat, average="macro", zero_division=0)), 4),
    roc_auc=round(float(roc_auc_score(y_te, p_te)), 4),
    average_precision=round(float(average_precision_score(y_te, p_te)), 4),
    prevalence=round(float(y_te.mean()), 4),
    confusion=dict(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1])),
)
print(json.dumps(metrics, indent=2))

# ---------------------------------------------------------------- explainer
BACKGROUND_N = 100
bg = shap.sample(X_tr, BACKGROUND_N, random_state=SEED)
explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional",
                               model_output="raw")
t0 = time.time()
sv_raw = np.array(explainer.shap_values(X_te))
sv_margin = sv_raw if sv_raw.ndim == 2 else sv_raw[:, :, 1]  # class-1 log-odds
sv_time = time.time() - t0
base_lo = float(np.atleast_1d(explainer.expected_value)[-1])
print(f"TreeExplainer (interventional, background n={BACKGROUND_N}, raw=log-odds) "
      f"ran in {sv_time:.1f}s; base value = {base_lo:.4f}")

margin_recon = base_lo + sv_margin.sum(axis=1)
margin_of_model = np.log(p_te / (1 - p_te))
resid = np.abs(margin_recon - margin_of_model)
prob_recon = 1.0 / (1.0 + np.exp(-margin_recon))
prob_resid = np.abs(prob_recon - p_te)

metrics_explainer = dict(
    explainer="shap.TreeExplainer",
    feature_perturbation="interventional",
    background=f"{BACKGROUND_N} random training rows (seed {SEED})",
    output_scale="log-odds of stroke = 1 (model margin)",
    positive_class_index=1,
    classes=model.classes_.tolist(),
    explained_set=f"all {len(X_te)} test rows",
    runtime_sec=round(sv_time, 1),
    base_value_log_odds=round(base_lo, 4),
    base_value_as_probability=round(float(1 / (1 + np.exp(-base_lo))), 4),
    additivity_max_abs_residual_log_odds=float(resid.max()),
    additivity_mean_abs_residual_log_odds=float(resid.mean()),
    additivity_probability_max_abs_residual=float(prob_resid.max()),
    tolerance=1e-6,
    note=("Additivity is checked in the log-odds space where the model is additive by "
          "construction; probability is obtained only afterwards through the logistic "
          "link 1/(1+exp(-m)), never by adding log-odds contributions to a probability."),
)
print(json.dumps(metrics_explainer, indent=2))

np.save(OUT / "shap_values_test_logodds.npy", sv_margin)
np.save(OUT / "test_pred_proba.npy", p_te)
X_te.to_csv(OUT / "X_test_encoded.csv")
d["raw"].loc[d["source_rows"]["test"]].to_csv(OUT / "X_test_raw.csv")
joblib.dump(model, OUT / "xgb_model.joblib")
save_json(dict(metrics=metrics, explainer=metrics_explainer,
               base_log_odds=base_lo, threshold=thr,
               shap_runtime_sec=round(sv_time, 1),
               total_runtime_sec=round(time.time() - t_start, 1)),
         OUT / "task2_metrics.json")
print("saved task2 outputs")
