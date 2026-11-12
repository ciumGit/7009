"""Task 3: local explanation for one held-out case.

Selection rule: among true positives in the explained held-out sample, take the
case with the median predicted probability. The rule is fixed before the
contributions are inspected, so the case is representative rather than the most
extreme one.

One-hot levels of the same predictor are recombined into a single contribution
(signed sum per case) before plotting; the force plot shows the eight largest
groups plus one remainder group so that labels stay readable.

Run from the repository root:  python analysis/task3_local.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

OUT, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)

meta = json.load(open(f"{OUT}/explainer_meta.json"))
base, thr, feats = meta["base_value"], meta["threshold"], meta["feature_names"]

Z = pd.read_csv(f"{OUT}/explained_features.csv")
SV = pd.read_csv(f"{OUT}/shap_values_class1.csv")
raw = pd.read_csv(f"{OUT}/explained_raw_rows.csv")

tp = raw[(raw["y_true"] == 1) & (raw["p_stroke"] >= thr)]
i = int((tp["p_stroke"] - tp["p_stroke"].median()).abs().idxmin())

GROUPS = ["age", "bmi", "avg_glucose_level", "hypertension", "heart_disease", "gender",
          "ever_married", "work_type", "Residence_type", "smoking_status"]
PRETTY = {"age": "Age", "bmi": "BMI", "avg_glucose_level": "Blood sugar",
          "hypertension": "High blood pressure", "heart_disease": "Heart disease",
          "ever_married": "Married", "work_type": "Job", "Residence_type": "Residence",
          "smoking_status": "Smoking", "gender": "Gender"}


def group_of(name):
    for g in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        if name.startswith(g + "_"):
            return g
    return name


def display_value(g):
    """Original-unit value shown under the force plot label."""
    r = raw.loc[i]
    if g == "age":
        return f"{r['age']:.0f}y"
    if g == "bmi":
        return f"{r['bmi']:.1f}"
    if g == "avg_glucose_level":
        return f"{r['avg_glucose_level']:.0f}"
    if g == "hypertension":
        return "yes" if int(r["hypertension"]) == 1 else "no"
    if g == "heart_disease":
        return "yes" if int(r["heart_disease"]) == 1 else "no"
    return str(r[g]).replace("_", " ")


# Signed grouped contributions for this case
gvals = {g: 0.0 for g in GROUPS}
for j, f in enumerate(feats):
    gvals[group_of(f)] += float(SV.iloc[i, j])

ordered = sorted(gvals.items(), key=lambda kv: abs(kv[1]), reverse=True)
top = ordered[:6]
rest = sum(v for _, v in ordered[6:])

names = [PRETTY[g] for g, _ in top] + ["10 remaining features"]
values = np.array([v for _, v in top] + [rest], dtype=float)
shown = np.array([[display_value(g) for g, _ in top] + ["summed"]])

# Readable grouped force plot
fig = plt.figure(figsize=(14, 2.9))
shap.force_plot(base, values.reshape(1, -1), shown, feature_names=names, matplotlib=True,
                show=False, text_rotation=0, plot_cmap=["#4C78A8", "#C0504D"])
plt.savefig(f"{FIG}/fig2_force_plot.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Contribution table (grouped, original units)
rows = []
for g, v in ordered:
    rows.append({"feature": PRETTY[g], "patient_value": display_value(g),
                 "shap_value": round(v, 4),
                 "direction": "increases stroke probability" if v > 0 else "decreases stroke probability"})
rows[-1]["note"] = "remaining features summed into the last rows" if False else ""
contrib = pd.DataFrame(rows)
contrib.to_csv(f"{OUT}/case_contributions.csv", index=False)

# Patient-friendly chart for slide 3 (zero-based bars, plain wording)
pat_labels = {"age": f"Age ({raw.loc[i,'age']:.0f} years)", "bmi": f"Body mass index ({raw.loc[i,'bmi']:.1f})",
              "avg_glucose_level": f"Blood sugar ({raw.loc[i,'avg_glucose_level']:.0f} mg/dL)",
              "hypertension": "High blood pressure" if int(raw.loc[i, "hypertension"]) == 1
              else "No high blood pressure",
              "heart_disease": "Heart disease" if int(raw.loc[i, "heart_disease"]) == 1
              else "No heart disease",
              "ever_married": "Married" if raw.loc[i, "ever_married"] == "Yes" else "Not married",
              "work_type": "Job: " + str(raw.loc[i, "work_type"]).replace("_", " ").lower(),
              "Residence_type": f"Living in a {str(raw.loc[i, 'Residence_type']).lower()} area",
              "smoking_status": "Smoking: " + str(raw.loc[i, "smoking_status"]),
              "gender": str(raw.loc[i, "gender"])}

o = sorted(sorted(gvals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6],
           key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(10.5, 5.0))
lab = [pat_labels[g] for g, _ in o]
val = [v for _, v in o]
colors = ["#C0504D" if v > 0 else "#4C78A8" for v in val]
ax.barh(lab, val, color=colors, height=0.58)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(min(0, min(val)) * 1.3 - 0.004, max(val) * 1.18)
ax.set_xlabel("How much this detail moved the score (probability points)")
ax.set_title("Which details moved my score up or down", fontsize=12, loc="left")
ax.text(0.99, 0.02, "red = raised the score, blue = lowered it", transform=ax.transAxes,
        ha="right", fontsize=8.5, color="grey")
for y, v in enumerate(val):
    ax.annotate(f"{v:+.2f}", (v, y), xytext=(7, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_patient_view.png", dpi=200, bbox_inches="tight")
plt.close(fig)

case = {
    "explained_row": i,
    "selection_rule": "true positive with median predicted probability in the explained held-out sample",
    "y_true": int(raw.loc[i, "y_true"]),
    "p_stroke": round(float(raw.loc[i, "p_stroke"]), 4),
    "threshold": thr,
    "case_type": "true positive",
    "base_value": round(base, 5),
    "sum_grouped_shap": round(float(sum(gvals.values())), 5),
    "reconstruction": round(base + float(sum(gvals.values())), 5),
    "reconstruction_residual": round(abs(base + float(sum(gvals.values())) - raw.loc[i, "p_stroke"]), 6),
    "grouped": {PRETTY[g]: round(v, 4) for g, v in ordered},
    "grouped_raw": {g: round(v, 4) for g, v in ordered},
    "raw_record": {k: (float(raw.loc[i, k]) if isinstance(raw.loc[i, k], (np.floating, float)) else raw.loc[i, k])
                   for k in ["gender", "age", "hypertension", "heart_disease", "ever_married",
                             "work_type", "Residence_type", "avg_glucose_level", "bmi",
                             "smoking_status"]},
}
json.dump(case, open(f"{OUT}/case_task3.json", "w"), indent=2, default=str)
print(json.dumps(case, indent=2, default=str))
print(contrib.to_string(index=False))
