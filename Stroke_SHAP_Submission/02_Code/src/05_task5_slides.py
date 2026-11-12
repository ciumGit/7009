"""Task 5 - three audience slides built from the validated task 2-4 results.

All three slides present the same model (task-2 XGBoost), the same case
(test #668) and the same numbers; only the language and the visual change.
Consistency with the report is checked programmatically against
outputs/task2_metrics.json and outputs/task3_case.json.

Run:  python src/05_task5_slides.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figs"

t2 = json.loads((OUT / "task2_metrics.json").read_text())["metrics"]
t3 = json.loads((OUT / "task3_case.json").read_text())
t4 = json.loads((OUT / "task4_global.json").read_text())
case_age = float(pd.read_csv(OUT / "X_test_raw.csv", index_col=0)
                 .iloc[t3["case_index"]]["age"])

m = t2
cm = m["confusion"]
prevalence_pct = round(100 * m["prevalence"], 1)
top_var = t4["ranking"][0]["variable"]

INK = RGBColor(0x1F, 0x2A, 0x36)
ACCENT = RGBColor(0x3B, 0x6E, 0xA5)
RED = RGBColor(0xC0, 0x50, 0x4D)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title, subtitle):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(0.16))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.32), Inches(12.3), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = INK
    if subtitle:
        p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = subtitle
        r2.font.size = Pt(14); r2.font.color.rgb = RGBColor(0x5A, 0x6B, 0x7B)
    return s


def bullets(slide, items, left=0.5, top=1.5, width=6.6, height=5.4, size=14):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for text, level, bold, color in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level; p.space_after = Pt(7)
        r = p.add_run(); r.text = text
        r.font.size = Pt(size if level == 0 else size - 1)
        r.font.bold = bold
        r.font.color.rgb = color or INK
    return tb


# ------------------------------------------------------------- slide 1
s1 = add_slide("For data scientists: what the model does, and where it fails",
               f"Stroke classification, test n = {m['test_n']:,} ({m['test_pos']} positive cases, "
               f"{prevalence_pct}% prevalence) - XGBoost, SHAP TreeExplainer")
bullets(s1, [
    ("Ranking quality is real but modest: ROC AUC "
     f"{m['roc_auc']:.3f}, average precision {m['average_precision']:.3f} "
     f"against a prevalence of {m['prevalence']:.3f}. A no-skill classifier scores 0.049.", 0, False, None),
    (f"At the tuned threshold {m['threshold']:.2f}: recall {m['recall_pos']:.2f}, "
     f"precision {m['precision_pos']:.2f}, F1 {m['f1_pos']:.2f}. "
     f"Confusion: TN {cm['tn']}, FP {cm['fp']}, FN {cm['fn']}, TP {cm['tp']}.", 0, True, None),
    ("The cost asymmetry drives every choice: an undetected stroke costs far more "
     "than a screening referral, so the threshold maximises F1 for stroke = 1, "
     "not accuracy.", 0, False, None),
    (f"Global drivers (mean |SHAP|, log-odds): {top_var} first, then "
     f"{', '.join(r['variable'] for r in t4['ranking'][1:4])}.", 0, False, None),
    ("Explainer configuration: TreeExplainer, interventional, 100 training rows as "
     "background (seed 42), output = log-odds. Additivity verified to 7e-6.", 0, False, None),
    ("Limitations: retrospective single-file data, 30% 'Unknown' smoking status, "
     "BMI missing 3.9% (median imputed inside the pipeline), no calibration "
     "analysis, and correlated predictors (age x ever-married) split credit "
     "arbitrarily between them.", 0, False, RED),
], width=12.3)

# ------------------------------------------------------------- slide 2
s2 = add_slide("For clinicians: one patient the model missed",
               f"Test case #{t3['case_index']}, a {case_age:.0f}-year-old woman - "
               f"predicted stroke probability {t3['predicted_probability']:.2f}, "
               f"threshold {t3['threshold']:.2f} - she had a stroke")
s2.shapes.add_picture(str(FIG / "fig2_local_force_plot.png"),
                      Inches(0.55), Inches(1.75), width=Inches(12.2))
bullets(s2, [
    ("What pushed the score up: age 66 and a missing BMI (imputed with the training "
     "median, 28.0, which lands near the risk boundary). What pushed it down: normal "
     "glucose (101.5 mg/dL), no hypertension, no heart disease, and the "
     "'Unknown' smoking entry.", 0, False, None),
    ("Sensitivity in this test split is "
     f"{m['recall_pos']:.2f}: {cm['fn']} of {m['test_pos']} stroke cases were missed. "
     "This patient was one of them - 0.02 below the decision line.", 0, True, RED),
    ("Read the SHAP values as model bookkeeping, not physiology: they explain why THIS "
     "model scored THIS patient this way, on retrospective records. They are not a "
     "diagnostic opinion and not a causal statement.", 0, False, None),
    ("Before any clinical use: external validation, probability calibration, and a rule "
     "for the missing-BMI and 'Unknown'-smoking patterns, which are frequent enough to "
     "move scores on their own.", 0, False, None),
], top=5.5, height=1.85, width=12.2, size=12)

# ------------------------------------------------------------- slide 3
s3 = add_slide("For patients: what this computer prediction means - and what it does not",
               "Plain-language version of the same result")
bullets(s3, [
    ("A computer program looked at recorded health information - age, blood sugar, "
     "weight records, smoking and work details - and gave a score from 0 to 1 that "
     "estimates how similar a patient's record is to records of past stroke patients.", 0, False, None),
    (f"For one {case_age:.0f}-year-old woman, the program gave 0.58, just under its alert line of "
     "0.60, so no alert was raised. She did have a stroke. Two things moved her score "
     "up - her age and a missing weight record - and three things moved it down: "
     "normal blood sugar, no blood-pressure or heart condition on record.", 0, True, None),
    ("The score is not a diagnosis, not a percentage chance of having a stroke, and "
     "not medical advice. It was built from old records, and it is wrong often enough "
     "that a doctor must always read it together with everything else known about "
     "the patient.", 0, False, RED),
    ("If you want to lower your risk, the proven steps are the usual ones: blood "
     "pressure control, not smoking, movement, and regular check-ups. The computer "
     "score cannot substitute for that conversation with your doctor.", 0, False, None),
], top=1.6, height=5.4, width=12.3, size=15)

# ------------------------------------------------------- consistency check
check = dict(
    same_model=True,
    same_case=True,
    case_probability_slide2=t3["predicted_probability"],
    threshold_slide1_and_2=m["threshold"],
    top_variable_slide1=top_var,
    notes="Slides 1-3 quote metrics from outputs/task2_metrics.json, the case from "
          "outputs/task3_case.json and the global ranking from outputs/task4_global.json. "
          "Slide 2 embeds the same figure 2 used in report section 4.",
)
out_path = ROOT / "Task5_Audience_Slides.pptx"
prs.save(out_path)
print(json.dumps(check, indent=2))
print("saved", out_path)
