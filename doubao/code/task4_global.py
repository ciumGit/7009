"""Task 4: global explanation with a SHAP summary (beeswarm) plot.

Run from the repository root:  python analysis/task4_global.py
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
meta = json.load(open(f"{OUT}/explainer_meta.json"))
feats = meta["feature_names"]

Z = pd.read_csv(f"{OUT}/explained_features.csv")
SV = pd.read_csv(f"{OUT}/shap_values_class1.csv").values
raw = pd.read_csv(f"{OUT}/explained_raw_rows.csv")

# Readable column labels, keeping the encoded level visible
def label(c):
    numeric = {"age": "Age", "bmi": "BMI", "avg_glucose_level": "Blood sugar",
               "hypertension": "High blood pressure", "heart_disease": "Heart disease"}
    if c in numeric:
        return numeric[c]
    for src, short in [("ever_married", "ever married"), ("work_type", "work type"),
                       ("Residence_type", "residence"), ("smoking_status", "smoking"),
                       ("gender", "gender")]:
        if c.startswith(src + "_"):
            return f"{short} = {c[len(src)+1:]}"
    return c


Zp = Z.copy()
Zp.columns = [label(c) for c in feats]

# ---------------- Figure 4a: beeswarm summary plot ----------------
plt.figure()
shap.summary_plot(SV, Zp, plot_type="dot", max_display=12, show=False,
                  color_bar_label="Feature value (low to high)", alpha=0.85)
plt.title("Global SHAP summary, class 1 (stroke), 300 held-out patients", fontsize=10)
plt.xlabel("SHAP value (change in predicted stroke probability)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_summary_beeswarm.png", dpi=220, bbox_inches="tight")
plt.close()

# ---------------- Grouped importance ----------------
GROUPS = {"age": "Age", "bmi": "BMI", "avg_glucose_level": "Blood sugar",
          "hypertension": "High blood pressure", "heart_disease": "Heart disease"}
for g in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    GROUPS[g] = {"gender": "Gender", "ever_married": "Marital status", "work_type": "Job type",
                 "Residence_type": "Residence", "smoking_status": "Smoking"}[g]


def group_of(name):
    for g in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        if name.startswith(g + "_"):
            return g
    return name


G = pd.DataFrame(0.0, index=range(len(SV)), columns=list(GROUPS))
for j, f in enumerate(feats):
    G[group_of(f)] += SV[:, j]

imp = pd.DataFrame({"group": list(GROUPS),
                    "mean_abs_shap": G.abs().mean(axis=0).values})
imp["pretty"] = imp["group"].map(GROUPS)
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp["share_pct"] = (imp["mean_abs_shap"] / imp["mean_abs_shap"].sum() * 100).round(1)

# Mean |SHAP| of the encoded columns, for the appendix
enc_imp = pd.DataFrame({"feature": [label(c) for c in feats],
                        "mean_abs_shap": np.abs(SV).mean(axis=0)}).sort_values(
    "mean_abs_shap", ascending=False).reset_index(drop=True)

# ---------------- Figure 4b: grouped importance bar ----------------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
head = imp.head(8)[::-1]
ax.barh(head["pretty"], head["mean_abs_shap"], color="#4C78A8", height=0.6)
ax.set_xlabel("Mean |SHAP| (grouped, probability scale)")
ax.set_title("Global importance after recombining encoded columns", fontsize=10)
for y, (v, n) in enumerate(zip(head["mean_abs_shap"], head["share_pct"])):
    ax.text(v + 0.004, y, f"{v:.3f}  ({n:.1f}%)", va="center", fontsize=8)
ax.set_xlim(0, head["mean_abs_shap"].max() * 1.22)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4b_grouped_importance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

summary = {
    "n_explained": int(len(SV)),
    "positive_in_sample": int(raw["y_true"].sum()),
    "base_value": meta["base_value"],
    "grouped_importance": imp.to_dict("records"),
    "encoded_importance_top10": enc_imp.head(10).to_dict("records"),
    "share_top3": round(float(imp["share_pct"].head(3).sum()), 1),
}
json.dump(summary, open(f"{OUT}/global_summary.json", "w"), indent=2)
imp.to_csv(f"{OUT}/global_importance_grouped.csv", index=False)
print(imp.to_string(index=False))
print(enc_imp.head(8).to_string(index=False))
