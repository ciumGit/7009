"""Task 4 - global explanation with a SHAP summary plot.

Explained set: the 1,022 test rows, class stroke = 1, log-odds scale.
Because the 21 model columns include 16 one-hot columns, the ranking table
aggregates each dummy block back to its parent variable: signed contributions
of all columns of one variable are summed per patient, and mean(|aggregated|)
is ranked. The beeswarm keeps the model-level columns (one dot per column);
the bar chart shows the aggregated per-variable view. Both are saved so the
report can quote one scale consistently.

Run:  python src/04_task4_global_summary.py
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
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prep import save_json  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figs"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

X_te = pd.read_csv(OUT / "X_test_encoded.csv", index_col=0)
sv = np.load(OUT / "shap_values_test_logodds.npy")
t2 = json.loads((OUT / "task2_metrics.json").read_text())

# ------------------------------------------------------- beeswarm (fig 3a)
plt.figure()
shap.summary_plot(sv, X_te, max_display=15, show=False, plot_size=(8.6, 6.4))
plt.gcf().axes[-1].set_ylabel("feature value\n(low -> high)", fontsize=9)
plt.xlabel("SHAP value (log-odds of stroke = 1)", fontsize=9)
plt.title("Figure 3a  SHAP summary plot - 1,022 test patients, stroke = 1 output",
          fontsize=10, pad=14)
plt.tight_layout()
plt.savefig(FIG / "fig3a_summary_beeswarm.png")
plt.close()
print("saved fig3a_summary_beeswarm.png")

# --------------------------------------------- mean |SHAP| bars (fig 3b)
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(-mean_abs)
ranked = [(X_te.columns[i], float(mean_abs[i])) for i in order]
plt.figure(figsize=(8.6, 4.6))
plt.barh([n for n, _ in reversed(ranked[:15])], [v for _, v in reversed(ranked[:15])],
         color="#3b6ea5")
plt.xlabel("mean |SHAP| (log-odds)")
plt.title("Figure 3b  Mean absolute SHAP value per encoded column", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "fig3b_mean_abs_bar.png")
plt.close()

# ------------------------------------- aggregated per-variable ranking (fig 3c)
groups = {}
for j, col in enumerate(X_te.columns):
    parent = col.split(" = ")[0] if " = " in col else col
    groups.setdefault(parent, []).append(j)
agg = pd.DataFrame({g: sv[:, idx].sum(axis=1) for g, idx in groups.items()})
agg_mean_abs = agg.abs().mean().sort_values(ascending=False)
plt.figure(figsize=(8.6, 4.2))
plt.barh(list(reversed(agg_mean_abs.index)), list(reversed(agg_mean_abs.values)), color="#c0504d")
plt.xlabel("mean |summed SHAP| per variable (log-odds)")
plt.title("Figure 3c  Global importance after merging one-hot blocks", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "fig3c_grouped_importance.png")
plt.close()
print("saved fig3b/fig3c")

# ---------------------------------------------------------- ranking table (5.2)
def pattern(col_or_var: str) -> str:
    a = agg if col_or_var in agg.columns else None
    if a is not None:
        v = a[col_or_var].values
        x = X_te[[c for c in X_te.columns if (c.split(" = ")[0] if " = " in c else c) == col_or_var]].sum(axis=1).values
    else:
        v = sv[:, X_te.columns.get_loc(col_or_var)]
        x = X_te[col_or_var].values
    if np.corrcoef(x, v)[0, 1] > 0:
        return "higher values push the output up"
    return "higher values push the output down"

table = []
for rank, var in enumerate(agg_mean_abs.index[:6], start=1):
    table.append({
        "rank": rank,
        "variable": var,
        "mean_abs_shap_log_odds": round(float(agg_mean_abs[var]), 4),
        "n_encoded_columns": len(groups[var]),
        "observed_pattern": pattern(var),
    })
print(json.dumps(table, indent=2))

# top variable directional detail for the narrative
top_var = agg_mean_abs.index[0]
idx = groups[top_var]
sub = pd.DataFrame(sv[:, idx], columns=[X_te.columns[j] for j in idx])
stats_top = {c: {"mean_pos": round(float(sub.loc[sub[c] > 0, c].mean()), 3) if (sub[c] > 0).any() else None,
                 "mean_neg": round(float(sub.loc[sub[c] < 0, c].mean()), 3) if (sub[c] < 0).any() else None,
                 "share_positive": round(float((sub[c] > 0).mean()), 3)}
             for c in sub.columns}
print(json.dumps(stats_top, indent=2))

save_json(dict(ranking=table, per_column_top10=[
    {"column": n, "mean_abs_shap_log_odds": round(v, 4)} for n, v in ranked[:10]
], grouped_stats_top_var={top_var: stats_top},
   explained_set="1022 test rows", output="log-odds of stroke = 1"),
   OUT / "task4_global.json")
print("saved task4_global.json")
