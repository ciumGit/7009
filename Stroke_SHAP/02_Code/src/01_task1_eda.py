"""Task 1 - exploratory data analysis.

from 00_data_exploration.py. 

  * matplotlib forced to the non-interactive Agg backend;
  * plt.show() replaced by plt.savefig() so figures reach outputs/figs/;
  * per-column histograms collected into one multi-panel figure (figure 1 of
    the report) instead of twelve separate windows;
  * descriptive statistics written to JSON so the numbers quoted in the report
    can be traced back to this run;
  * the pre-processing preview now uses prep.build_preprocessor() so that the
    preview matches the leak-free pipeline used for modelling.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prep  # noqa: E402

DATA = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
FIG = ROOT / "outputs" / "figs"
OUT = ROOT / "outputs"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=0.85)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

df = prep.load_raw(DATA)
n_rows, n_cols = df.shape
stats: dict = {"shape": [n_rows, n_cols]}

# ---------------------------------------------------------------- basic checks
stats["duplicate_ids"] = int(df["id"].duplicated().sum())
stats["missing"] = {
    c: {"n": int(df[c].isna().sum()), "pct": round(100 * df[c].isna().mean(), 2)}
    for c in df.columns
    if df[c].isna().any()
}
stats["stroke_counts"] = {str(k): int(v) for k, v in df["stroke"].value_counts().sort_index().items()}
stats["stroke_positive_pct"] = round(100 * df["stroke"].mean(), 2)

num_desc = {}
for c in ["age", "avg_glucose_level", "bmi"]:
    s = df[c].dropna()
    num_desc[c] = {
        "mean": round(float(s.mean()), 2),
        "sd": round(float(s.std()), 2),
        "median": round(float(s.median()), 2),
        "q1": round(float(s.quantile(0.25)), 2),
        "q3": round(float(s.quantile(0.75)), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
    }
stats["numeric"] = num_desc

cat_desc = {}
for c in prep.CATEGORICAL:
    vc = df[c].value_counts()
    cat_desc[c] = {str(k): {"n": int(v), "pct": round(100 * v / n_rows, 2)} for k, v in vc.items()}
stats["categorical"] = cat_desc

# positive rate by subgroup - used in section 2.3 of the report
by_age = df.assign(age_band=pd.cut(df["age"], [-0.1, 17, 39, 59, 200],
                                  labels=["<18", "18-39", "40-59", "60+"]))
stats["stroke_rate_by_age_band"] = {
    str(k): {"n": int(v), "pos": int(by_age.loc[by_age.age_band == k, "stroke"].sum()),
             "rate_pct": round(100 * by_age.loc[by_age.age_band == k, "stroke"].mean(), 2)}
    for k, v in by_age["age_band"].value_counts().sort_index().items()
}
stats["stroke_rate_by_hypertension"] = {
    str(k): round(100 * g["stroke"].mean(), 2) for k, g in df.groupby("hypertension")
}
stats["stroke_rate_by_heart_disease"] = {
    str(k): round(100 * g["stroke"].mean(), 2) for k, g in df.groupby("heart_disease")
}
stats["stroke_rate_by_smoking"] = {
    str(k): round(100 * g["stroke"].mean(), 2) for k, g in df.groupby("smoking_status")
}
stats["stroke_rate_by_glucose_quartile"] = {
    str(k): round(100 * g["stroke"].mean(), 2)
    for k, g in df.groupby(pd.qcut(df["avg_glucose_level"], 4, duplicates="drop"), observed=True)
}

# ------------------------------------------------------------------- figure 1
BLUE, RED = "#3b6ea5", "#c0504d"
fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.2))

# a) target balance
ax = axes[0, 0]
counts = df["stroke"].value_counts().sort_index()
bars = ax.bar(["No stroke (0)", "Stroke (1)"], counts.values, color=[BLUE, RED], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v}\n({100*v/n_rows:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, counts.max() * 1.22)
ax.set_ylabel("Patients (n)")
ax.set_title("(a) Target balance: 1 positive per 20 negatives", loc="left", fontsize=10)

# b) missingness
ax = axes[0, 1]
miss = (df.isna().sum() / n_rows * 100).sort_values(ascending=True)
miss = miss[miss > 0]
labels = [m.replace("_", " ").capitalize() for m in miss.index]
ax.barh(labels, miss.values, color="#d98880", height=0.32, edgecolor="#a94442")
ax.set_ylim(-0.8, 0.8)
ax.text(miss.values[0] + 0.15, 0, f"{miss.values[0]:.2f}%  (n={df[miss.index[0]].isna().sum()})",
        va="center", fontsize=9)
ax.set_xlim(0, 6.2)
ax.set_xlabel("Missing values (%)")
ax.set_title("(b) Only BMI is incomplete", loc="left", fontsize=10)

# c) age by outcome
ax = axes[0, 2]
sns.kdeplot(data=df, x="age", hue="stroke", fill=True, common_norm=False,
            palette={0: BLUE, 1: RED}, alpha=0.45, linewidth=1.1, ax=ax, legend=False)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density (normalised within class)")
ax.set_title("(c) Stroke cases are concentrated above 60", loc="left", fontsize=10)
ax.text(0.03, 0.92, "blue = no stroke, red = stroke", transform=ax.transAxes, fontsize=8)

# d) glucose by outcome
ax = axes[1, 0]
sns.boxplot(data=df, x="stroke", y="avg_glucose_level", hue="stroke",
            palette={0: BLUE, 1: RED}, legend=False, width=0.45, ax=ax, fliersize=1.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No stroke", "Stroke"])
ax.set_xlabel(""); ax.set_ylabel("Average glucose (mg/dL)")
ax.set_title("(d) Glucose: higher median and wider spread in stroke", loc="left", fontsize=10)

# e) bmi by outcome
ax = axes[1, 1]
sns.boxplot(data=df.dropna(subset=["bmi"]), x="stroke", y="bmi", hue="stroke",
            palette={0: BLUE, 1: RED}, legend=False, width=0.45, ax=ax, fliersize=1.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No stroke", "Stroke"])
ax.set_xlabel(""); ax.set_ylabel("BMI (kg/m$^2$)")
ax.set_title("(e) BMI: near-identical distributions", loc="left", fontsize=10)
ax.text(0.02, 0.92, f"{df['bmi'].isna().sum()} BMI values missing (excluded)",
        transform=ax.transAxes, fontsize=8)

# f) comorbidity rates
ax = axes[1, 2]
cond = pd.DataFrame({
    "Hypertension": df.groupby("hypertension")["stroke"].mean() * 100,
    "Heart disease": df.groupby("heart_disease")["stroke"].mean() * 100,
})
x = np.arange(2)
ax.bar(x - 0.17, cond.loc[0].values, 0.32, label="Condition absent", color=BLUE)
ax.bar(x + 0.17, cond.loc[1].values, 0.32, label="Condition present", color=RED)
for i, v in enumerate(cond.loc[0].values):
    ax.text(i - 0.17, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
for i, v in enumerate(cond.loc[1].values):
    ax.text(i + 0.17, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cond.columns)
ax.set_ylabel("Stroke rate (%)"); ax.legend(fontsize=7, frameon=False)
ax.set_title("(f) Both comorbidities raise the stroke rate", loc="left", fontsize=10)

# g) smoking status composition
ax = axes[2, 0]
smoke_lab = {"never smoked": "Never smoked", "formerly smoked": "Formerly smoked",
             "smokes": "Smokes", "Unknown": "Unknown"}
df_g = df.assign(smoking_status=df["smoking_status"].map(smoke_lab))
order = ["Never smoked", "Unknown", "Formerly smoked", "Smokes"]
sns.countplot(data=df_g, y="smoking_status", hue="stroke", order=order,
              palette={0: BLUE, 1: RED}, ax=ax)
ax.set_ylabel(""); ax.set_xlabel("Patients (n)")
ax.set_title("(g) 'Unknown' smoking status is a 30% block", loc="left", fontsize=10)
ax.legend(["No stroke", "Stroke"], fontsize=7, frameon=False, loc="lower right")

# h) numeric correlation
ax = axes[2, 1]
corr = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.72}, annot_kws={"size": 7}, ax=ax)
ax.set_title("(h) Pearson correlation, numeric variables", loc="left", fontsize=10)
ax.tick_params(labelsize=7)

# i) age vs glucose
ax = axes[2, 2]
ax.scatter(df.loc[df.stroke == 0, "age"], df.loc[df.stroke == 0, "avg_glucose_level"],
           s=5, c=BLUE, alpha=0.22, label="No stroke", linewidths=0)
ax.scatter(df.loc[df.stroke == 1, "age"], df.loc[df.stroke == 1, "avg_glucose_level"],
           s=11, c=RED, alpha=0.85, label="Stroke", linewidths=0.2, edgecolors="white")
ax.set_xlabel("Age (years)"); ax.set_ylabel("Average glucose (mg/dL)")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("(i) Positive cases scatter, they do not cluster", loc="left", fontsize=10)

fig.suptitle(
    "Figure 1  EDA panel - stroke prediction dataset (n = 5,110; 249 stroke cases, 4.87%)",
    fontsize=12, y=1.005,
)
fig.tight_layout()
fig.savefig(FIG / "fig1_eda_panel.png")
plt.close(fig)

# ------------------------------------------------- pre-processing preview (repo step)
loaded = prep.prepare(DATA)
stats["n_features_after_encoding"] = int(loaded["X_train"].shape[1])
stats["feature_names"] = loaded["feature_names"]
stats["split_sizes"] = {
    "train": int(len(loaded["y_train"])),
    "val": int(len(loaded["y_val"])),
    "test": int(len(loaded["y_test"])),
    "train_pos": int(loaded["y_train"].sum()),
    "val_pos": int(loaded["y_val"].sum()),
    "test_pos": int(loaded["y_test"].sum()),
}
prep.save_json(stats, OUT / "eda_stats.json")

print(json.dumps({k: stats[k] for k in
                  ["shape", "duplicate_ids", "missing", "stroke_counts",
                   "stroke_positive_pct", "n_features_after_encoding", "split_sizes"]}, indent=2))
print("\nfeature names:", len(loaded["feature_names"]))
print("\nsaved:", FIG / "fig1_eda_panel.png", "and", OUT / "eda_stats.json")
