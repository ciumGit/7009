"""Task 1: exploratory data analysis for the stroke dataset.

Run from the repository root:  python analysis/task1_eda.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results"
FIG = "figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DATA = "data/healthcare-dataset-stroke-data.csv"

df = pd.read_csv(DATA)
summary = {}
summary["n_rows"], summary["n_cols"] = df.shape
summary["columns"] = list(df.columns)
summary["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

# Duplicate records and identifiers
summary["duplicate_rows"] = int(df.duplicated().sum())
summary["duplicate_ids"] = int(df["id"].duplicated().sum())

# Missing values
miss = df.isna().sum()
summary["missing"] = {c: int(v) for c, v in miss.items() if v > 0}
summary["missing_pct"] = {c: round(float(v) / len(df) * 100, 2) for c, v in miss.items() if v > 0}

# Target
counts = df["stroke"].value_counts().sort_index()
summary["stroke_counts"] = {int(k): int(v) for k, v in counts.items()}
summary["stroke_prevalence"] = round(float(counts.get(1, 0)) / len(df) * 100, 2)

# Numeric summaries overall and by outcome
for col in ["age", "avg_glucose_level", "bmi"]:
    s = df[col]
    summary[f"{col}_overall"] = {
        "mean": round(float(s.mean()), 2),
        "sd": round(float(s.std()), 2),
        "median": round(float(s.median()), 2),
        "q1": round(float(s.quantile(0.25)), 2),
        "q3": round(float(s.quantile(0.75)), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
    }
    summary[f"{col}_by_stroke"] = {
        str(int(k)): {"median": round(float(g.median()), 2), "mean": round(float(g.mean()), 2), "n": int(len(g))}
        for k, g in df.groupby("stroke")[col]
    }

# Categorical distributions and stroke rates
cat_summary = {}
for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    tab = df.groupby(col)["stroke"].agg(["count", "mean"])
    cat_summary[col] = {
        str(k): {"n": int(r["count"]), "pct": round(float(r["count"]) / len(df) * 100, 2),
                 "stroke_rate": round(float(r["mean"]) * 100, 2)}
        for k, r in tab.iterrows()
    }
summary["categorical"] = cat_summary

# Plausibility flags
summary["bmi_gt_60"] = int((df["bmi"] > 60).sum())
summary["age_under_1"] = int((df["age"] < 1).sum())
summary["glucose_gt_200"] = int((df["avg_glucose_level"] > 200).sum())

with open(f"{OUT}/eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ---------------- Figure 1: six-panel EDA overview ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))

# (a) class imbalance
ax = axes[0, 0]
bars = ax.bar(["No stroke (0)", "Stroke (1)"], counts.values, color=["#8FA9C4", "#C0504D"], width=0.55)
ax.set_title("(a) Target class distribution")
ax.set_ylabel("Patients")
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v}\n({v/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 5600)
ax.spines[["top", "right"]].set_visible(False)

# (b) age distribution by outcome
ax = axes[0, 1]
bins = range(0, 85, 5)
ax.hist([df.loc[df.stroke == 0, "age"], df.loc[df.stroke == 1, "age"]], bins=bins,
        stacked=False, color=["#8FA9C4", "#C0504D"], label=["No stroke", "Stroke"], rwidth=0.9)
ax.set_title("(b) Age by outcome")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Patients")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# (c) glucose distribution by outcome
ax = axes[0, 2]
ax.hist([df.loc[df.stroke == 0, "avg_glucose_level"], df.loc[df.stroke == 1, "avg_glucose_level"]],
        bins=30, color=["#8FA9C4", "#C0504D"], label=["No stroke", "Stroke"])
ax.set_title("(c) Average glucose by outcome")
ax.set_xlabel("Average glucose level (mg/dL)")
ax.set_ylabel("Patients")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# (d) BMI with missing values flagged
ax = axes[1, 0]
ax.hist(df["bmi"].dropna(), bins=30, color="#7F9E7F")
ax.set_title("(d) BMI distribution (201 missing, 3.9%)")
ax.set_xlabel("BMI (kg/m2)")
ax.set_ylabel("Patients")
ax.axvline(df["bmi"].median(), color="black", ls="--", lw=1)
ax.text(df["bmi"].median() + 1, ax.get_ylim()[1] * 0.85, f"median {df['bmi'].median():.1f}",
        fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# (e) smoking status: counts and stroke rate
ax = axes[1, 1]
order = ["never smoked", "formerly smoked", "smokes", "Unknown"]
cnt = [cat_summary["smoking_status"][k]["n"] for k in order]
rate = [cat_summary["smoking_status"][k]["stroke_rate"] for k in order]
x = range(len(order))
ax.bar(x, cnt, color="#8FA9C4", width=0.55, label="Patients")
ax.set_xticks(list(x))
ax.set_xticklabels(["never\nsmoked", "formerly\nsmoked", "smokes", "Unknown"], fontsize=8)
ax.set_title("(e) Smoking status and stroke rate")
ax.set_ylabel("Patients")
ax2 = ax.twinx()
ax2.plot(list(x), rate, "o-", color="#C0504D", lw=1.2, ms=5, label="Stroke rate")
ax2.set_ylabel("Stroke rate (%)", color="#C0504D")
ax2.tick_params(axis="y", colors="#C0504D", labelsize=8)
ax2.set_ylim(0, 10)
ax.spines[["top"]].set_visible(False);
ax2.spines[["top"]].set_visible(False)

# (f) age band vs stroke rate
ax = axes[1, 2]
band = pd.cut(df["age"], bins=[0, 18, 35, 50, 65, 85],
              labels=["<18", "18-34", "35-49", "50-64", "65+"])
r = df.groupby(band, observed=True)["stroke"].agg(["mean", "count"])
ax.bar(range(len(r)), r["mean"] * 100, color="#C0504D", width=0.55)
ax.set_xticks(range(len(r)))
ax.set_xticklabels(r.index, fontsize=8)
for i, (m, c) in enumerate(zip(r["mean"] * 100, r["count"])):
    ax.text(i, m + 0.3, f"{m:.1f}%\n(n={c})", ha="center", va="bottom", fontsize=7)
ax.set_title("(f) Stroke rate by age band")
ax.set_ylabel("Stroke rate (%)")
ax.set_ylim(0, max(r["mean"] * 100) * 1.35)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(f"{FIG}/fig1_eda_panel.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(json.dumps(summary, indent=2))
print("saved", f"{FIG}/fig1_eda_panel.png")
