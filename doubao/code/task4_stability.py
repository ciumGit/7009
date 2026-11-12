"""Stability check: repeat the Task 2 explanations with a second background sample.

Compares the grouped global ranking and the local contributions of the Section 4 case
under two 50-row training backgrounds (seeds 2021 and 777).

Run from the repository root:  python analysis/task4_stability.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

SEED = 2021
bundle = joblib.load("models/task2_pipeline.joblib")
model, pre = bundle["model"], bundle["preprocessor"]

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                            random_state=SEED)
cols = [c.replace("num__", "").replace("cat__", "") for c in pre.get_feature_names_out()]
Z_tr = pd.DataFrame(pre.transform(X_tr), columns=cols)
Z_te = pd.DataFrame(pre.transform(X_te), columns=cols)

sample_idx = Z_te.sample(n=300, random_state=SEED).index
Z_sub = Z_te.loc[sample_idx].reset_index(drop=True)
case_pos = 268  # row of the Section 4 case inside the 300-row sample

GROUPS = {"age": "Age", "bmi": "BMI", "avg_glucose_level": "Blood sugar",
          "hypertension": "High blood pressure", "heart_disease": "Heart disease",
          "gender": "Gender", "ever_married": "Marital status", "work_type": "Job type",
          "Residence_type": "Residence", "smoking_status": "Smoking"}


def group_of(name):
    for g in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        if name.startswith(g + "_"):
            return g
    return name


def grouped(sv, feats):
    out = {}
    for j, f in enumerate(feats):
        out[group_of(f)] = out.get(group_of(f), 0.0) + float(sv[j])
    return out


results = {}
for seed in (2021, 777):
    bg = shap.sample(Z_tr, 50, random_state=seed)
    ex = shap.TreeExplainer(model, data=bg, model_output="probability",
                            feature_perturbation="interventional")
    sv = ex.shap_values(Z_sub, check_additivity=False)
    sv1 = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    feats = list(Z_sub.columns)

    G = pd.DataFrame([grouped(row, feats) for row in sv1])
    imp = G.abs().mean().sort_values(ascending=False)
    local = grouped(sv1[case_pos], feats)
    results[seed] = {
        "base_value": round(float(np.ravel(ex.expected_value)[1]), 4),
        "top5": [GROUPS[k] for k in imp.index[:5]],
        "mean_abs": {GROUPS[k]: round(float(v), 4) for k, v in imp.items()},
        "case": {GROUPS[k]: round(float(v), 4) for k, v in local.items()},
        "case_output": round(float(model.predict_proba(Z_sub.iloc[[case_pos]])[:, 1][0]), 4),
    }

a, b = results[2021], results[777]
sign_flips = [k for k in a["case"] if np.sign(a["case"][k]) != np.sign(b["case"][k])]
rank_shift = {k: a["top5"].index(k) - b["top5"].index(k) if k in b["top5"] else None
              for k in a["top5"]}
summary = {
    "background_A_seed": 2021, "background_B_seed": 777,
    "base_A": a["base_value"], "base_B": b["base_value"],
    "top5_A": a["top5"], "top5_B": b["top5"],
    "top5_overlap": len(set(a["top5"]) & set(b["top5"])),
    "case_contributions_A": a["case"],
    "case_contributions_B": b["case"],
    "case_sign_flips": sign_flips,
    "case_output_A": a["case_output"], "case_output_B": b["case_output"],
    "mean_abs_comparison": {k: [a["mean_abs"][k], b["mean_abs"][k]] for k in a["mean_abs"]},
}
json.dump(summary, open("results/stability.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
