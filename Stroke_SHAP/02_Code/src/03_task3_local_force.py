"""Task 3 - local explanation of one held-out patient with a force plot.

Case selection rule, declared before looking at any explanation: among the
test rows the model misses (predicted probability below the 0.60 threshold,
true stroke = 1), take the false negative with the highest predicted
probability - the patient the model came closest to catching. A miss, not a
flattering success, is deliberate: the report template asks for a discussion
of what a false negative means, and picking a high-confidence correct
prediction would hide exactly that.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prep import SEED, load_raw, save_json  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

X_te = pd.read_csv(OUT / "X_test_encoded.csv", index_col=0)
raw_te = pd.read_csv(OUT / "X_test_raw.csv", index_col=0)
p_te = np.load(OUT / "test_pred_proba.npy")
sv = np.load(OUT / "shap_values_test_logodds.npy")
t2 = json.loads((OUT / "task2_metrics.json").read_text())
thr = t2["threshold"]
base = t2["base_log_odds"]
model = joblib.load(OUT / "xgb_model.joblib")

y_te = raw_te["stroke"].astype(int).values
pred = (p_te >= thr).astype(int)
case_type = []
for truth, p in zip(y_te, pred):
    case_type.append({(0, 0): "true negative", (0, 1): "false positive",
                      (1, 0): "false negative", (1, 1): "true positive"}[(truth, p)])
case_types = np.array(case_type)

# ------------------------------------------------------------------ pick case
fn_idx = np.where(case_types == "false negative")[0]
order = fn_idx[np.argsort(-p_te[fn_idx])]
case_i = int(order[0])
print(f"test false negatives: {len(fn_idx)}; chosen row {case_i} "
      f"(highest predicted probability among misses)")
print(f"true={y_te[case_i]} predicted={pred[case_i]} p={p_te[case_i]:.4f} type={case_types[case_i]}")

row = X_te.iloc[case_i]
raw_row = raw_te.iloc[case_i]
contrib = sv[case_i]
print("\ncontributions (log-odds):")
for name, c in sorted(zip(X_te.columns, contrib), key=lambda t: -abs(t[1])):
    print(f"  {name:28s} {c:+.4f}   value={row[name]}")

# ------------------------------------------------------------ force plot (fig 2)
margin_case = float(base + contrib.sum())

def display_label(name: str) -> str:
    """Patient-level label; one-hot columns show their own level and presence."""
    if " = " in name:
        col, level = name.split(" = ", 1)
        state = "present" if row[name] > 0.5 else "absent"
        return f"{col} = {level} ({state})"
    src = name
    if src == "age":
        return f"age = {raw_row['age']:.0f} years"
    if src == "bmi":
        return "BMI missing (median imputed)" if pd.isna(raw_row["bmi"]) \
            else f"BMI = {raw_row['bmi']:.1f}"
    if src == "avg_glucose_level":
        return f"glucose = {raw_row['avg_glucose_level']:.1f} mg/dL"
    if src == "hypertension":
        return f"hypertension = {int(raw_row['hypertension'])}"
    if src == "heart_disease":
        return f"heart disease = {int(raw_row['heart_disease'])}"
    return f"{src} = {row[name]:.2f}"

labels = [display_label(c) for c in X_te.columns]
disp = contrib.copy()
disp[np.abs(disp) < 0.01] = 0.0          # group negligible terms to keep labels legible
shap.force_plot(
    base, disp, np.array(labels, dtype=object),
    matplotlib=True, show=False, text_rotation=-12, link="identity",
)
fig = plt.gcf()
fig.set_size_inches(13.0, 4.8)
plt.gca().set_xlabel("")
fig.suptitle(
    f"Figure 2  Local SHAP force plot - test case #{case_i}, a false negative\n"
    f"base value {base:.3f} log-odds -> model output {margin_case:.3f} log-odds "
    f"(p = {p_te[case_i]:.3f}); decision threshold {thr:.2f}",
    fontsize=10, y=1.04,
)
plt.tight_layout(rect=(0, 0.02, 1, 0.94))
plt.savefig(FIG / "fig2_local_force_plot.png")
plt.close()
print("saved fig2_local_force_plot.png")

# waterfall as a readable supplement
expl = shap.Explanation(values=contrib, base_values=base, data=row.values,
                        feature_names=list(X_te.columns))
fig = plt.figure()
ax = shap.plots.waterfall(expl, max_display=12, show=False)
ax.set_title(f"Supplement to figure 2 - case #{case_i}, log-odds decomposition")
plt.gcf().set_size_inches(8.2, 6.4)
plt.tight_layout()
plt.savefig(FIG / "fig2b_local_waterfall.png")
plt.close()

# ----------------------------------------------------- contribution table (4.2)
def patient_value(name: str) -> str:
    return display_label(name)


ordc = np.argsort(-np.abs(contrib))
top = [(X_te.columns[k], float(contrib[k])) for k in ordc[:3]]
table_rows = []
for name, c in top:
    table_rows.append({"feature": name, "patient_value": patient_value(name),
                       "shap_log_odds": round(c, 4),
                       "direction": "pushes toward stroke" if c > 0 else "pushes away from stroke"})
rest = float(contrib.sum() - sum(c for _, c in top))
table_rows.append({"feature": "remaining 18 features (combined)",
                   "patient_value": "-", "shap_log_odds": round(rest, 4),
                   "direction": "pushes toward stroke" if rest > 0 else "pushes away from stroke"})
table_rows.append({"feature": "base value (background average)",
                   "patient_value": "-", "shap_log_odds": round(base, 4),
                   "direction": "-"})
save_json(dict(case_index=case_i, case_type=case_types[case_i],
               true_label=int(y_te[case_i]), predicted_label=int(pred[case_i]),
               predicted_probability=round(float(p_te[case_i]), 4),
               threshold=thr, base_log_odds=base,
               model_output_log_odds=round(float(base + contrib.sum()), 4),
               n_false_negatives=int(len(fn_idx)),
               omitted_below_threshold_log_odds=round(float(contrib[np.abs(contrib) < 0.01].sum()), 4),
               label_grouping_note=("contributions with |phi| < 0.01 log-odds are omitted from the "
                                    "force-plot labels, so the f(x) printed on the plot (0.35) differs "
                                    "from the exact model output (0.337) by their combined -0.015"),
               contributions=[{"feature": n, "value": patient_value(n),
                               "shap_log_odds": round(float(c), 4)}
                              for n, c in sorted(zip(X_te.columns, contrib), key=lambda t: -abs(t[1]))],
               table=table_rows),
         OUT / "task3_case.json")
print(json.dumps(table_rows, indent=2))
print("saved task3_case.json")
