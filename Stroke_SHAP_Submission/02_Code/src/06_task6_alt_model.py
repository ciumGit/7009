"""Task 6 - explaining a different black-box model with KernelSHAP.

Task 2 used a gradient-boosted tree with TreeExplainer. Task 6 switches to an
RBF-kernel SVM - a genuinely different function class, no tree structure for
TreeExplainer to exploit - and explains it with KernelSHAP, the model-agnostic
route the report template suggests for this case.

Design choices recorded for section 7.2:
  * identical split and identical pre-processing as task 2; the background is
    a smaller draw of 25 training rows (seed 42) because KernelSHAP cost grows
    with the background size, and the difference is reported where it matters;
  * KernelSHAP explains predict_proba directly, so its output scale is the
    probability of stroke = 1 - NOT the log-odds scale of task 2. SHAP
    magnitudes from the two explainers are therefore never compared directly;
  * the global summary is computed on a fixed subsample of 150 test rows
    (seed 42) because KernelSHAP cost grows linearly with the explained set;
  * nsamples = 300 coalitions per row (above the 2*M minimum for 21 features)
    and the reconstruction residual is recorded;
  * the explainer wraps only the stroke = 1 probability, so SHAP values come
    back as a single (n, features) array in probability units.

Run:  python src/06_task6_alt_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prep import SEED, prepare, save_json  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figs"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

DATA = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
GLOBAL_N = 150
t_start = time.time()

d = prepare(DATA)
X_tr, y_tr = d["X_train"], d["y_train"]
X_va, y_va = d["X_val"], d["y_val"]
X_te, y_te = d["X_test"], d["y_test"]

# ------------------------------------------------------------------ tune SVM
rows = []
for C in (0.5, 1.0, 3.0, 10.0):
    for gamma in ("scale", 0.05, 0.02):
        m = SVC(C=C, gamma=gamma, kernel="rbf", probability=True,
                class_weight="balanced", random_state=SEED)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        rows.append((C, gamma, roc_auc_score(y_va, p), average_precision_score(y_va, p)))
        print(f"C={C} gamma={gamma}  val AUC {rows[-1][2]:.4f}  val AP {rows[-1][3]:.4f}")
best_C, best_gamma, _, _ = max(rows, key=lambda r: r[2] + r[3])
print(f"selected C={best_C}, gamma={best_gamma}")

svm = SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True,
          class_weight="balanced", random_state=SEED)
svm.fit(X_tr, y_tr)

p_val = svm.predict_proba(X_va)[:, 1]
thr_grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
f1s = [f1_score(y_va, (p_val >= t).astype(int), zero_division=0) for t in thr_grid]
thr = float(thr_grid[int(np.argmax(f1s))])

p_te = svm.predict_proba(X_te)[:, 1]
y_hat = (p_te >= thr).astype(int)
cm = confusion_matrix(y_te, y_hat)
metrics = dict(
    model="SVC (RBF kernel, probability via internal Platt scaling)",
    params={"C": best_C, "gamma": str(best_gamma), "kernel": "rbf",
            "class_weight": "balanced", "random_state": SEED},
    threshold=thr,
    test_n=int(len(y_te)), test_pos=int(y_te.sum()),
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

# ------------------------------------------------------------- KernelSHAP
bg = shap.sample(X_tr, 100, random_state=SEED)          # same rows as task 2
bg = shap.sample(X_tr, 25, random_state=SEED)
explainer = shap.KernelExplainer(lambda Z: svm.predict_proba(Z)[:, 1], bg,
                                 link="identity")
rng = np.random.RandomState(SEED)
global_idx = rng.choice(len(X_te), size=GLOBAL_N, replace=False)
X_glob = X_te.iloc[global_idx]

t0 = time.time()
sv_glob = explainer.shap_values(X_glob, nsamples=300, l1_reg=0, silent=True)
sv_time = time.time() - t0
sv_pos = np.squeeze(np.array(sv_glob))
assert sv_pos.ndim == 2 and sv_pos.shape == (len(X_glob), X_glob.shape[1]), sv_pos.shape
base_prob = float(np.squeeze(explainer.expected_value))
print(f"KernelSHAP global ({GLOBAL_N} rows, nsamples=300) ran in {sv_time:.1f}s; "
      f"base value P[stroke=1] on background = {base_prob:.4f}")

prob_recon = base_prob + sv_pos.sum(axis=1)
resid = np.abs(prob_recon - svm.predict_proba(X_glob)[:, 1])
print(f"max reconstruction residual {resid.max():.2e}, mean {resid.mean():.2e}")

# local explanation of the SAME false-negative case as task 3
t3 = json.loads((OUT / "task3_case.json").read_text())
case_i = t3["case_index"]
row = X_te.iloc[[case_i]]
t1 = time.time()
_c = np.array(explainer.shap_values(row, nsamples=600, l1_reg=0, silent=True))
_c = np.squeeze(_c)
sv_case = _c[0] if _c.ndim == 2 else _c
case_time = time.time() - t1
case_p = float(svm.predict_proba(row)[:, 1][0])
case_recon = base_prob + sv_case.sum()
print(f"case #{case_i}: SVM p = {case_p:.4f} (task-2 model gave "
      f"{t3['predicted_probability']:.4f}); reconstruction residual "
      f"{abs(case_recon - case_p):.2e}")

# ------------------------------------------------------------------ figures
def display_label(name: str, enc_row) -> str:
    if " = " in name:
        col, level = name.split(" = ", 1)
        return f"{col} = {level} ({'present' if enc_row[name] > 0.5 else 'absent'})"
    if name == "age":
        return f"age = {raw_case['age']:.0f} years"
    if name == "bmi":
        return "BMI missing (median imputed)" if pd.isna(raw_case["bmi"]) \
            else f"BMI = {raw_case['bmi']:.1f}"
    if name == "avg_glucose_level":
        return f"glucose = {raw_case['avg_glucose_level']:.1f} mg/dL"
    if name == "hypertension":
        return f"hypertension = {int(raw_case['hypertension'])}"
    if name == "heart_disease":
        return f"heart disease = {int(raw_case['heart_disease'])}"
    return name

raw_case = pd.read_csv(OUT / "X_test_raw.csv", index_col=0).iloc[case_i]
case_labels = np.array([display_label(c, row.iloc[0]) for c in X_te.columns], dtype=object)
disp_case = sv_case.copy()
disp_case[np.abs(disp_case) < 0.002] = 0.0   # group negligible terms on the probability scale

# fig 4 - local force plot for the SVM, probability scale
plt.figure()
shap.force_plot(base_prob, disp_case, case_labels,
                matplotlib=True, show=False, text_rotation=-12, link="identity")
fig = plt.gcf()
fig.set_size_inches(13.0, 4.8)
plt.gca().set_xlabel("")
fig.suptitle(
    f"Figure 4  KernelSHAP force plot - same test case #{case_i}, SVM model\n"
    f"base value {base_prob:.3f} -> model output {sv_case.sum() + base_prob:.3f} "
    f"(probability scale); contributions below 0.002 grouped",
    fontsize=10, y=1.04,
)
plt.tight_layout(rect=(0, 0.02, 1, 0.94))
plt.savefig(FIG / "fig4_alt_local_force.png")
plt.close()

# fig 5 - global beeswarm, probability scale
plt.figure()
shap.summary_plot(sv_pos, X_glob, max_display=15, show=False, plot_size=(8.6, 6.4))
plt.xlabel("SHAP value (probability of stroke = 1)", fontsize=9)
plt.title(f"Figure 5  KernelSHAP summary plot - {GLOBAL_N} test patients, SVM model",
          fontsize=10, pad=14)
plt.tight_layout()
plt.savefig(FIG / "fig5_alt_global_summary.png")
plt.close()
print("saved fig4 and fig5")

mean_abs_svm = pd.Series(np.abs(sv_pos).mean(axis=0), index=X_te.columns).sort_values(ascending=False)
top3 = [c for c in mean_abs_svm.index[:3]]
groups = {}
for col in X_te.columns:
    groups.setdefault(col.split(" = ")[0] if " = " in col else col, []).append(col)
agg = pd.DataFrame({g: sv_pos[global_idx if False else slice(None), [X_te.columns.get_loc(c) for c in cols]].sum(axis=1)
                    for g, cols in groups.items()})
agg_mean_abs = agg.abs().mean().sort_values(ascending=False)

save_json(dict(
    metrics=metrics,
    explainer=dict(
        explainer="shap.KernelExplainer",
        model_output="predict_proba (probability of stroke = 1)",
        background="25 training rows (seed 42); task 2 used 100 rows - both drawn from the same training split",
        nsamples="300 (global) / 600 (single case), no L1 regularisation",
        link="identity",
        global_set=f"{GLOBAL_N} test rows sampled with seed {SEED}",
        runtime_global_sec=round(sv_time, 1),
        runtime_local_sec=round(case_time, 1),
        base_value_probability=round(base_prob, 4),
        additivity_max_abs_residual_probability=round(float(resid.max()), 8),
        additivity_mean_abs_residual_probability=round(float(resid.mean()), 10),
        tolerance=0.002,
    ),
    case=dict(index=case_i,
              svm_probability=round(case_p, 4),
              svm_threshold=thr,
              task2_probability=t3["predicted_probability"],
              task2_threshold=t3["threshold"],
              reconstruction=round(float(case_recon), 5)),
    top3_columns=top3,
    grouped_ranking=[{"variable": v, "mean_abs_shap_probability": round(float(x), 5)}
                     for v, x in agg_mean_abs.head(6).items()],
    total_runtime_sec=round(time.time() - t_start, 1),
), OUT / "task6_metrics.json")
print("saved task6_metrics.json")
