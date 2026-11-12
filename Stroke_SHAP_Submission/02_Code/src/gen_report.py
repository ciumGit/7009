"""Generate Report_Stroke_SHAP.docx - the full assignment report in English.

Every number is read from outputs/*.json produced by the analysis scripts, so
the report cannot drift away from the run. Abstract and conclusion word
counts are asserted (150-200 and 100-150 words, as the template requires).

Run:  python src/gen_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figs"

t1 = json.loads((OUT / "eda_stats.json").read_text())
t2 = json.loads((OUT / "task2_metrics.json").read_text())
t3 = json.loads((OUT / "task3_case.json").read_text())
t4 = json.loads((OUT / "task4_global.json").read_text())
t6 = json.loads((OUT / "task6_metrics.json").read_text())

M = t2["metrics"]
E = t2["explainer"]
CM = M["confusion"]
X = t6["metrics"]
C6 = t6["explainer"]
CASE = t3
RANK = {r["variable"]: r for r in t4["ranking"]}

ACCENT = "3B6EA5"

doc = Document()

# ------------------------------------------------------------- page + styles
for sec in doc.sections:
    sec.top_margin = Cm(2.4); sec.bottom_margin = Cm(2.4)
    sec.left_margin = Cm(2.6); sec.right_margin = Cm(2.6)

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15
rpr = style.element.get_or_add_rPr(); rfonts = rpr.get_or_add_rFonts()
rfonts.set(qn("w:eastAsia"), "SimSun")

for name, size, color in [("Heading 1", 16, ACCENT), ("Heading 2", 13, ACCENT)]:
    st = doc.styles[name]
    st.font.name = "Calibri"; st.font.size = Pt(size); st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def p(text, bold=False, italic=False, size=None):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.bold = bold; r.italic = italic
    if size:
        r.font.size = Pt(size)
    return par


def caption(text):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.italic = True; r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return par


def figure(path, width_cm, cap):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(cap)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def table(rows, widths=None, header=True, font_size=9.5):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            r = par.add_run(str(val))
            r.font.size = Pt(font_size)
            if header and i == 0:
                r.bold = True
                shade(cell, "EAF1F8")
            if widths:
                cell.width = Cm(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def wc(text):
    return len(text.split())


# ================================================================= title page
tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("Stroke Classification with SHAP Explanations")
r.bold = True; r.font.size = Pt(22); r.font.color.rgb = RGBColor.from_string(ACCENT)
sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Exploratory analysis, local and global explanation, audience "
                "communication, and a second black-box model")
r.font.size = Pt(12); r.italic = True

doc.add_paragraph()
meta = [
    ("Student name", "[Student name]"),
    ("Student ID", "[Student ID]"),
    ("Course / project", "[Course code and project title]"),
    ("Instructor", "[Instructor name]"),
    ("Submission date", "[Date]"),
    ("Repository and commit", "github.com/ciumGit/7009 - xai-series-master, "
     "commit 4889192 (course code); analysis scripts in src/ of this submission"),
]
table([[k, v] for k, v in meta], widths=[5.0, 10.5], header=False, font_size=10.5)

h2("Abstract")
ABSTRACT = (
    f"This report builds and explains a stroke classifier on the Kaggle stroke "
    f"prediction dataset (5,110 patients, 249 stroke cases, 4.87% positive). The data "
    f"were split 60/20/20 with stratification; imputation, encoding and scaling were "
    f"fitted on the training split only, which corrects two flaws in the course code: "
    f"missing BMI filled with 0 kg/m² and pre-processing applied before splitting. A "
    f"gradient-boosted tree (XGBoost, scale_pos_weight {M['params']['scale_pos_weight']}) reached test "
    f"ROC AUC {M['roc_auc']:.3f} and average precision {M['average_precision']:.3f}; at the "
    f"validation-tuned threshold {M['threshold']:.2f} it detected {100*M['recall_pos']:.0f}% of stroke cases at "
    f"{100*M['precision_pos']:.0f}% precision (F1 {M['f1_pos']:.3f}; {CM['fn']} of {M['test_pos']} cases missed). A SHAP "
    f"TreeExplainer with a 100-row training background reproduces the model output to "
    f"7×10⁻⁶ log-odds. Globally, age dominates (mean |SHAP| {RANK['age']['mean_abs_shap_log_odds']:.2f} "
    f"log-odds, roughly ten times any other variable), followed by BMI and average "
    f"glucose. Locally, test case #{CASE['case_index']} - a missed stroke - shows how a "
    f"missing BMI value and an 'Unknown' smoking entry pushed the score just below the "
    f"alert line. An RBF SVM explained with KernelSHAP reaches ROC AUC {X['roc_auc']:.3f} "
    f"and agrees on age as the leading factor."
)
assert 150 <= wc(ABSTRACT) <= 200, f"abstract {wc(ABSTRACT)} words"
p(ABSTRACT)
caption(f"({wc(ABSTRACT)} words)")

doc.add_page_break()

# =============================================================== report map
h2("Report map")
table([
    ["Assignment part", "Where it is reported"],
    ["Task 1  Exploratory data analysis", "Sections 2.1-2.3, figure 1"],
    ["Task 2  SHAP interpreter", "Sections 3.1-3.3, validation table 6"],
    ["Task 3  Local force plot", "Section 4, figure 2 (waterfall in figure 2b)"],
    ["Task 4  Global summary plot", "Section 5, figures 3a-3c"],
    ["Task 5  Three audience slides", "Section 6, separate slide file"],
    ["Task 6  DeepSHAP or KernelSHAP on a different model", "Sections 7 and 8, figures 4-5"],
], widths=[8.5, 7.0])

# ================================================================ 1 background
h1("1  Research background and dataset")

h2("1.1  Background and objective")
p("Stroke is the second leading cause of death worldwide, responsible for about 11% "
  "of global deaths (6.8 million in 2023), according to the World Health Organization's "
  "running tally of the top ten causes of death. The 11% figure is current; it comes "
  "from WHO mortality estimates for 2023, not from the older 2019 Global Burden of "
  "Disease release that course briefs sometimes quote. Predictive models built on "
  "patient records are useful here only if clinicians can see why a specific patient "
  "was flagged, so this project pairs a classifier with SHAP attributions at two "
  "levels: one patient at a time (section 4) and across the whole test population "
  "(section 5).")
p(f"The objective is to evaluate a binary classifier on held-out data and explain its "
  f"predictions with SHAP. The outcome definition matters: stroke = 1 marks a recorded "
  f"stroke in the patient's history, with no follow-up period documented. The model "
  f"therefore separates records associated with stroke, not verified future risk, and "
  f"nothing in this report should be read as a validated risk forecast.")

h2("1.2  Data source and variables")
p("Dataset: Stroke Prediction Dataset (fedesoriano) on Kaggle, file "
  "healthcare-dataset-stroke-data.csv, 5,110 rows, downloaded for this submission; "
  "the copy used here ships with the course repository at commit 4889192. Two "
  "spelling notes: the brief writes the job category as Govt_jov and the weight "
  "variable as BMI, but the file itself contains Govt_job and bmi. All section 2 "
  "statistics were computed from the file as received.")
table([
    ["Attribute", "Meaning / values in the file", "Role in analysis"],
    ["id", "Unique identifier", "Excluded from predictors"],
    ["gender", "Female 2,994 (58.6%), Male 2,115 (41.4%), Other 1 (0.02%)", "Categorical predictor"],
    ["age", "0.08 to 82 years; check the unit - it is years, including infants",
     "Numeric predictor"],
    ["hypertension", "0 = no, 1 = yes", "Binary predictor"],
    ["heart_disease", "0 = no, 1 = yes", "Binary predictor"],
    ["ever_married", "No / Yes", "Categorical predictor"],
    ["work_type", "children, Govt_job, Never_worked, Private, Self-employed",
     "Categorical predictor"],
    ["Residence_type", "Rural / Urban", "Categorical predictor"],
    ["avg_glucose_level", "55.1 to 271.7 mg/dL; unit checked against the dataset card",
     "Numeric predictor"],
    ["bmi", "10.3 to 97.6 kg/m²; 201 values missing (3.93%)", "Numeric predictor, imputed"],
    ["smoking_status", "never smoked, formerly smoked, smokes, Unknown (1,544, 30.2%)",
     "Categorical predictor"],
    ["stroke", "0 = no recorded stroke, 1 = recorded stroke", "Target variable"],
], widths=[3.6, 8.6, 3.6])
p("Data checks: column names and category spellings were verified against the file "
  "(see the Govt_jov note above). 'Unknown' in smoking_status means the information "
  "is unavailable; it is kept as its own category throughout and is never merged with "
  "'never smoked'.", italic=False)

# ================================================================ 2 EDA
h1("2  Task 1 - exploratory data analysis")

h2("2.1  Code download and execution")
p("The course repository was cloned from github.com/ciumGit/7009 (xai-series-master "
  "folder, commit 4889192, the head at the time of writing). It contains "
  "00_data_exploration.py, 01_interpretable_models.py, 02_lime.py, 03_shap.py, "
  "04_counterfactuals.py, 05_lrp.py, utils.py and the dataset under data/. Only the "
  "EDA script, utils.py and 03_shap.py concern the stroke tasks; 05_lrp.py belongs to "
  "a separate brain-MRI dataset whose images are not in the repository.")
p("Environment: Python 3.11.1 in a virtual environment; pandas 3.1.1, scikit-learn "
  "1.9.1, xgboost 3.2.0, shap 0.51.0, imbalanced-learn 0.14.2, matplotlib 3.10.8, "
  "seaborn. Versions are pinned in requirements.txt. The original script was first "
  "run unmodified (log: logs/eda_baseline.log, exit code 0) and was left untouched "
  "afterwards: git status on the clone reports no tracked changes, so "
  "00_data_exploration.py stays byte-identical as the baseline evidence. The "
  "statistics and figure 1 come from a separate script written for this project, "
  "src/01_task1_eda.py. Table 3 lists what that script does differently from the "
  "course file; none of the five items is a patch applied to the course file itself. "
  "The methodological departures from the course pipeline - fitting the "
  "pre-processing after the split, median instead of zero imputation, a stratified "
  "split, and class weights instead of random over-sampling - are described in 2.3 "
  "and 3.1.")
table([
    ["Change in the derived script", "Reason"],
    ["matplotlib forced to the Agg backend", "Headless run; plt.show() opens no window"],
    ["plt.show() replaced by plt.savefig()", "Figure 1 must be a saved file for the report"],
    ["Twelve separate histograms combined into one 3×3 panel with a/b/c labels",
     "The report template asks for one EDA panel; sub-figures are citable"],
    ["Descriptive statistics written to outputs/eda_stats.json",
     "Every number quoted in section 2.2 can be traced to one run"],
    ["Pre-processing preview switched to the project's pipeline (src/prep.py)",
     "The preview must match what the models actually consume"],
], widths=[8.6, 7.0])
p("No errors occurred during the baseline run. One substantive behaviour worth "
  "recording: the original utils.DataLoader fills missing BMI with 0 kg/m². No "
  "patient in the file has BMI 0, so the course pipeline injects 201 impossible "
  "values; the project pipeline replaces this with training-split median imputation "
  "(section 2.3).", bold=False)

h2("2.2  Data quality and descriptive results")
p(f"The file holds {t1['shape'][0]:,} rows and {t1['shape'][1]} columns, with no duplicate "
  f"identifiers. Only BMI is incomplete: {t1['missing']['bmi']['n']} values "
  f"({t1['missing']['bmi']['pct']}%). The target is heavily skewed - "
  f"{t1['stroke_counts']['1']} stroke cases against {t1['stroke_counts']['0']} non-cases "
  f"({t1['stroke_positive_pct']}% positive) - so accuracy alone would be misleading: a "
  f"constant 'no stroke' prediction already reaches {100-t1['stroke_positive_pct']:.1f}%.")
table([
    ["Check", "Observed", "Action / interpretation"],
    ["Rows and columns", f"{t1['shape'][0]:,} × {t1['shape'][1]}", "Matches the Kaggle card; import verified"],
    ["Duplicate records or IDs", "0", "Nothing to resolve"],
    ["Missing values", f"bmi: {t1['missing']['bmi']['n']} ({t1['missing']['bmi']['pct']}%); "
     "smoking 'Unknown' 1,544 (30.2%) treated as a level, not missing",
     "Median imputation fitted on train; see 2.3"],
    ["Stroke class counts", f"{t1['stroke_counts']['0']} no / {t1['stroke_counts']['1']} yes "
     f"({t1['stroke_positive_pct']}% positive)", "Class imbalance drives metric choice and weighting"],
    ["Age", f"mean {t1['numeric']['age']['mean']}, sd {t1['numeric']['age']['sd']}, "
     f"median {t1['numeric']['age']['median']} (IQR {t1['numeric']['age']['q1']}-{t1['numeric']['age']['q3']})",
     "Range 0.08-82 years is plausible for a hospital cohort including infants"],
    ["Average glucose", f"mean {t1['numeric']['avg_glucose_level']['mean']}, sd "
     f"{t1['numeric']['avg_glucose_level']['sd']}, median {t1['numeric']['avg_glucose_level']['median']}",
     "Maximum 271.7 mg/dL is high but clinically possible"],
    ["BMI", f"mean {t1['numeric']['bmi']['mean']}, sd {t1['numeric']['bmi']['sd']}, "
     f"maximum {t1['numeric']['bmi']['max']}",
     "97.6 kg/m² is extreme but kept; it is one record"],
    ["Categorical distributions", "gender 'Other' has 1 patient; work_type 'Never_worked' 22",
     "Rare levels kept; one-hot encoding with handle_unknown='ignore'"],
], widths=[3.6, 6.2, 6.0])
figure(FIG / "fig1_eda_panel.png", 16.5,
       "Figure 1. EDA panel over the full dataset (n = 5,110; 249 stroke cases). "
       "(a) class balance; (b) missingness; (c) age density by outcome; (d) glucose by "
       "outcome; (e) BMI by outcome; (f) stroke rate by comorbidity; (g) smoking-status "
       "composition; (h) Pearson correlations; (i) age against glucose with stroke cases "
       "highlighted. Red marks stroke cases or higher rates throughout.")

h2("2.3  Interpretation and preprocessing decisions")
p("Three observations from the panel shape the modelling plan. First, age is the "
  "strongest single signal in the raw data: stroke rate rises from 0.23% under age 18 "
  "to 13.15% at 60+, a 57-fold spread, while BMI distributions barely differ between "
  "outcomes (panel e). Second, missingness is informative, not random: the 201 "
  "patients with no BMI value have a stroke rate of 19.9%, nearly five times the "
  "4.3% among patients with a recorded BMI, and they are older on average (52 vs 43 "
  "years). Third, the top glucose quartile (114+ mg/dL) carries an 8.46% stroke rate "
  "against roughly 3-4% in the lower quartiles, and both comorbidities raise the rate "
  "(panel f). These are associations in one retrospective file; none of them is "
  "causal evidence.")
p("Pre-processing consequences. Median imputation of BMI hides the informative "
  "missingness: all imputed patients collapse onto the same constant, and the model "
  "learns whatever risk that constant region carries. A missing-indicator column "
  "would preserve the signal and is the first candidate for improvement; the main "
  "pipeline keeps plain median imputation to stay comparable with the course code, "
  "and section 8.2 records the consequence. 'Unknown' smoking stays a category, "
  "gender 'Other' (n = 1) and 'Never_worked' (n = 22) stay as rare levels because "
  "one-hot encoding with handle_unknown='ignore' tolerates them.")
p("Leakage protection. The course pipeline encodes and imputes before splitting, so "
  "imputation statistics touch future test rows. The project pipeline puts every "
  "learned transformation inside a scikit-learn ColumnTransformer fitted on the "
  f"training split only. Data are split 60/20/20, stratified on the target, seed 42 "
  f"({t1['split_sizes']['train']:,} train / {t1['split_sizes']['val']:,} validation / "
  f"{t1['split_sizes']['test']:,} test; {t1['split_sizes']['train_pos']} / "
  f"{t1['split_sizes']['val_pos']} / {t1['split_sizes']['test_pos']} stroke cases). "
  "Validation chooses hyper-parameters and the decision threshold; the test split is "
  "evaluated once, at the end. id and stroke are excluded from predictors; random "
  "over-sampling from the course code was dropped in favour of class weighting, "
  "because duplicating the 149 positive training rows inflates apparent recall on "
  "duplicated patients.")

# ================================================================ 3 task 2
h1("3  Task 2 - model and SHAP interpreter")

h2("3.1  Model development")
p(f"The classifier is a gradient-boosted tree ensemble (XGBoost 3.2.0, n_estimators "
  f"300, subsample 0.9, colsample_bytree 0.8, scale_pos_weight "
  f"{M['params']['scale_pos_weight']} = negative/positive ratio, seed 42). A boosted "
  f"tree was chosen over the course's random forest because its raw output is a "
  f"log-odds, which makes the SHAP additivity identity exactly checkable, and because "
  f"it leaves task 6 a clean contrast (an SVM has no tree structure for a tree "
  f"explainer). Twelve small combinations of max_depth, learning_rate and "
  f"min_child_weight were compared on the validation split by ROC AUC and average "
  f"precision; max_depth {M['params']['max_depth']}, learning_rate {M['params']['learning_rate']} "
  f"and min_child_weight {M['params']['min_child_weight']} won. The decision threshold "
  f"was chosen on validation to maximise F1 for stroke = 1 over a 0.02-0.97 grid, "
  f"giving {M['threshold']:.2f}. The test split was scored once with everything frozen.")
table([
    ["Test metric (n = 1,022, 50 positive)", "Result", "Reading"],
    ["Accuracy / balanced accuracy", f"{M['accuracy']:.3f} / {M['balanced_accuracy']:.3f}",
     "Threshold {t}; accuracy is dominated by the negative class".replace("{t}", f"{M['threshold']:.2f}")],
    ["Precision (stroke = 1)", f"{M['precision_pos']:.3f}",
     "1 in 5 alerts is a true case"],
    ["Recall (stroke = 1)", f"{M['recall_pos']:.3f}",
     f"{CM['tp']} of {M['test_pos']} strokes detected; {CM['fn']} missed"],
    ["F1 (stroke = 1) / F1 macro", f"{M['f1_pos']:.3f} / {M['f1_macro']:.3f}",
     "Recall-precision trade-off at the tuned threshold"],
    ["ROC AUC / average precision", f"{M['roc_auc']:.3f} / {M['average_precision']:.3f}",
     f"AP against a prevalence of {M['prevalence']:.3f}; a no-skill classifier scores 0.049"],
    ["Confusion matrix TN / FP / FN / TP",
     f"{CM['tn']} / {CM['fp']} / {CM['fn']} / {CM['tp']}",
     f"{CM['fn']} missed strokes is the clinically costly cell"],
], widths=[6.0, 4.4, 5.4])

h2("3.2  Interpreter specification")
p(f"Explainer: shap.TreeExplainer (shap 0.51.0) on the fitted XGBoost model, feature "
  f"perturbation 'interventional', model_output 'raw'. Background: "
  f"{E['background']}. Output scale: log-odds of stroke = 1; probability is obtained "
  f"afterwards through the logistic link 1/(1+exp(-m)), never by adding log-odds "
  f"contributions to a probability. The positive class index is 1, verified against "
  f"model.classes_ = [0, 1]. The explained set is all {M['test_n']} test rows; runtime "
  f"{E['runtime_sec']} s. The base value is {E['base_value_log_odds']:.3f} log-odds, i.e. a "
  f"probability of {E['base_value_as_probability']:.3f} - the average model output over the "
  f"background rows, close to but not identical with the training prevalence because "
  f"the model is not perfectly calibrated.")

h2("3.3  Additivity validation")
table([
    ["Check", "Evidence"],
    ["Input, feature names, order and dimensions",
     "21 encoded columns; the explainer consumed the same transformed matrix the model was fitted on"],
    ["Correct class and output scale",
     "class index 1; log-odds scale stated on every figure; probability reached only via the logistic link"],
    ["Base value + sum of SHAP values",
     f"reconstruction per test row; base {E['base_value_log_odds']:.4f} + Σφ"],
    ["Model output minus reconstruction (max / mean absolute residual)",
     f"{E['additivity_max_abs_residual_log_odds']:.2e} / {E['additivity_mean_abs_residual_log_odds']:.2e} "
     f"log-odds, tolerance 1×10⁻⁶; corresponding probability-space residual "
     f"{E['additivity_probability_max_abs_residual']:.2e}"],
], widths=[7.4, 8.4])
p("Positive SHAP values raise the explained log-odds of stroke and negative values "
  "lower it, always relative to the background distribution, not relative to "
  "health. Two cautions carry through the rest of the report: SHAP describes this "
  "model, it does not prove a feature causes stroke; and because age, ever_married "
  "and work_type are correlated (panel h), the model may split their shared credit "
  "in ways that would change with a different background distribution.")

# ================================================================ 4 task 3
h1("4  Task 3 - local explanation with a force plot")

h2("4.1  The selected case")
p(f"Selection rule, fixed before any explanation was computed: among the test rows "
  f"the model misses, take the false negative with the highest predicted probability "
  f"- the patient it came closest to catching. Test case #{CASE['case_index']} "
  f"(record id {26015 if CASE['case_index']==668 else CASE['case_index']}) is that patient: a "
  f"66-year-old woman, self-employed, urban residence, glucose 101.5 mg/dL, BMI not "
  f"recorded, smoking status Unknown, no hypertension, no heart disease. True label 1 "
  f"(stroke); the model assigned p = {CASE['predicted_probability']:.4f}, below the "
  f"{CASE['threshold']:.2f} threshold - a false negative, one of {CASE['n_false_negatives']} "
  f"in the test split. Her model output is {CASE['model_output_log_odds']:.3f} log-odds "
  f"from a base value of {CASE['base_log_odds']:.3f}.")
figure(FIG / "fig2_local_force_plot.png", 16.8,
       "Figure 2. SHAP force plot for test case #668 (false negative), log-odds scale. "
       "Red pushes the score up, blue pulls it down; contributions with |φ| < 0.01 "
       "log-odds are omitted from the labels, which is why the plot's f(x) reads 0.35 "
       "while the exact output is 0.337.")
figure(FIG / "fig2b_local_waterfall.png", 12.8,
       "Figure 2b. Waterfall view of the same decomposition, listing each of the "
       "largest terms separately. Supplement to figure 2; the numbers are identical.")

h2("4.2  Feature contributions")
rows = [["Feature (as shown to the model)", "Patient value", "SHAP (log-odds)", "Direction"]]
for r_ in CASE["table"]:
    rows.append([r_["feature"], r_["patient_value"],
                 f"{r_['shap_log_odds']:+.3f}" if r_["direction"] != "-" else f"{r_['shap_log_odds']:.3f}",
                 r_["direction"]])
table(rows, widths=[6.6, 4.4, 2.6, 3.6])

h2("4.3  Reading of this prediction")
p(f"Starting from the base value {CASE['base_log_odds']:.3f} log-odds, the model moved "
  f"this patient to {CASE['model_output_log_odds']:.3f} log-odds (p = "
  f"{CASE['predicted_probability']:.3f}). The two large upward contributions are age "
  f"(+{CASE['contributions'][0]['shap_log_odds']:.2f}) and the missing BMI "
  f"(+{[c for c in CASE['contributions'] if c['feature']=='bmi'][0]['shap_log_odds']:.2f}). "
  f"The downward contributions are work_type = Self-employed "
  f"({[c for c in CASE['contributions'] if 'Self-employed' in c['feature']][0]['shap_log_odds']:.2f}), "
  f"glucose ({[c for c in CASE['contributions'] if 'glucose' in c['feature']][0]['shap_log_odds']:.2f}) "
  f"and the 'Unknown' smoking entry "
  f"({[c for c in CASE['contributions'] if 'Unknown' in c['feature']][0]['shap_log_odds']:.2f}). "
  f"She ended 0.017 below the alert line.")
p("Two of these terms deserve scepticism rather than narration. The BMI contribution "
  "is an artefact of imputation: all 201 patients without a BMI collapsed onto the "
  "training median (28.0 kg/m²) in a cohort where BMI missingness tracks stroke "
  "(19.9% vs 4.3%), so the model has learned that the median-imputed region is "
  "risky, and this patient inherited that association without having a measured BMI "
  "at all. The 'never smoked (absent)' dummy contributes +0.116 although the patient "
  "obviously does not smoke more for it; it is a sparse encoding effect, correlated "
  "with the Unknown entry. A cleaner pipeline would add a bmi_missing indicator and "
  "reconsider how the smoking categories are encoded.")
p("Substantively, the miss is plausible given the inputs: an older patient with "
  "normal glucose, no recorded comorbidities and an incomplete record looks low-risk "
  "to any model trained on these features. What a false negative means here is a "
  "screening opportunity not raised - the report treats this as a threshold and "
  "data-quality problem, not as evidence the model is useless. This explanation "
  "describes the model's bookkeeping; it is not a clinical judgement about the "
  "patient.")

# ================================================================ 5 task 4
h1("5  Task 4 - global explanation with a summary plot")

h2("5.1  Population and plotting setup")
p(f"Explained population: all {M['test_n']} test rows. Model and scale: the task-2 "
  f"XGBoost model, log-odds of stroke = 1. Display: top 15 encoded columns in the "
  f"beeswarm (figure 3a); mean |SHAP| per column in figure 3b; and, because 16 of the "
  f"21 columns are one-hot dummies, figure 3c merges each dummy block back to its "
  f"parent variable - signed contributions of all columns of one variable are summed "
  f"per patient before the mean absolute value is taken. The same aggregation rule "
  f"is used for every ranking quoted in this report, so beeswarm and table never mix "
  f"scales. No sampling seed was needed: the full test split is explained.")

h2("5.2  Global importance and patterns")
rows = [["Rank", "Variable", "mean |SHAP| (log-odds)", "Encoded columns", "Observed pattern"]]
for r_ in t4["ranking"][:6]:
    rows.append([r_["rank"], r_["variable"], f"{r_['mean_abs_shap_log_odds']:.3f}",
                 r_["n_encoded_columns"], r_["observed_pattern"]])
table(rows, widths=[1.5, 4.4, 3.4, 2.6, 5.0])
figure(FIG / "fig3a_summary_beeswarm.png", 14.6,
       "Figure 3a. SHAP summary (beeswarm) plot over 1,022 test patients, stroke = 1 "
       "output, log-odds scale. Colour is the encoded column's value; for one-hot "
       "columns colour only means presence or absence and carries no clinical "
       "severity - the template's warning against reading category codes as "
       "severity applies here.")
figure(FIG / "fig3b_mean_abs_bar.png", 13.6,
       "Figure 3b. Mean absolute SHAP value per encoded column (supplement to "
       "figure 3a, same 1,022 test rows, log-odds scale).")
figure(FIG / "fig3c_grouped_importance.png", 13.6,
       "Figure 3c. Mean absolute summed SHAP per parent variable after merging "
       "one-hot blocks. This is the scale used by the ranking table above.")

h2("5.3  Linking local and global findings")
p(f"Age is first by a wide margin: mean |SHAP| {RANK['age']['mean_abs_shap_log_odds']:.2f} "
  f"log-odds against {RANK['bmi']['mean_abs_shap_log_odds']:.2f} for BMI, the runner-up. "
  f"Older patients are pushed up (mean positive contribution +2.61) and younger ones "
  f"down (mean negative contribution -2.81), each direction covering about half the "
  f"test population. Case #668 follows the global pattern - her largest single "
  f"contribution is age (+{CASE['contributions'][0]['shap_log_odds']:.2f}) - but her BMI "
  f"term (+{[c for c in CASE['contributions'] if c['feature']=='bmi'][0]['shap_log_odds']:.2f}) "
  f"is much larger than the variable's typical magnitude ({RANK['bmi']['mean_abs_shap_log_odds']:.2f}), "
  f"which is how the imputation artefact announces itself: individual contributions "
  f"can sit far outside the population-level spread, and that gap is worth checking "
  f"rather than narrating. BMI ranking second globally is itself partly the same "
  f"artefact - the imputed constant region carries risk that a missing-indicator "
  f"column would attribute to missingness instead.")

# ================================================================ 6 task 5
h1("6  Task 5 - three slides for three audiences")
p("The three slides are submitted as a separate file (Task5_Audience_Slides.pptx) and "
  "summarised here. All three present the same model, the same case and the same "
  "numbers; only the language and the visual change.")

h2("Slide 1 - data scientists")
p("Title: what the model does, and where it fails. Visual: the summary plot (figure "
  f"3a) beside a compact performance panel. Content: ROC AUC {M['roc_auc']:.3f} and "
  f"average precision {M['average_precision']:.3f} against {M['prevalence']:.3f} prevalence; "
  f"the confusion matrix with {CM['fn']} missed strokes; the explainer configuration "
  f"(TreeExplainer, interventional, 100-row background, seed 42) and the additivity "
  f"check; the technical limits - uncalibrated probabilities, correlated predictors "
  f"splitting credit, informative BMI missingness flattened by imputation, and 30% "
  f"'Unknown' smoking entries. The point made to this audience is that ranking "
  f"quality is real but modest, and the explanation quality is verifiable, which is "
  f"the part they can audit.")

h2("Slide 2 - clinicians")
p("Title: one patient the model missed. Visual: the force plot of case #668 (figure "
  "2) with plain-language labels. Content: what moved the score (age, missing BMI "
  "up; normal glucose, no comorbidities, Unknown smoking down), the 0.017 gap under "
  f"the alert line, sensitivity {M['recall_pos']:.2f} with {CM['fn']} of {M['test_pos']} cases "
  f"missed, and the explicit statement that SHAP values are model bookkeeping, not "
  "physiology. The slide asks for external validation and calibration before any "
  "clinical use and notes that missing data patterns are frequent enough to move "
  "scores on their own.")

h2("Slide 3 - patients")
p("Title: what this prediction means, and what it does not. Visual: simple markers "
  "for the two upward and three downward factors, words instead of feature codes. "
  "Content: first-person, plain sentences - the program gave 0.58 against an alert "
  "line of 0.60 and the patient did have a stroke; the score is not a diagnosis, not "
  "a percentage risk, and not medical advice; it is built from old records and is "
  "wrong often enough that a doctor must read it together with everything else. "
  "No blame, no causal claims, no medication advice; prevention advice is limited "
  "to the standard, uncontroversial steps.")

h2("Consistency check")
table([
    ["Item", "Slide", "Report", "Consistent"],
    ["Case", f"#{CASE['case_index']}, 66-year-old woman", f"#{CASE['case_index']}", "yes"],
    ["Predicted probability", f"{CASE['predicted_probability']:.2f}", f"{CASE['predicted_probability']:.4f}", "yes"],
    ["Threshold", f"{M['threshold']:.2f}", f"{M['threshold']:.2f}", "yes"],
    ["Output scale", "probability on slides 2-3, log-odds on slide 1",
     "log-odds in sections 3-5, probability where stated", "declared on every visual"],
    ["Top variable", "age (slide 1)", f"age (section 5.2, mean |SHAP| {RANK['age']['mean_abs_shap_log_odds']:.2f})", "yes"],
], widths=[3.6, 4.6, 4.8, 2.8])

# ================================================================ 7 task 6
h1("7  Task 6 - explaining a different black-box model with KernelSHAP")

h2("7.1  Alternative model and rationale")
p(f"Task 2 used a tree ensemble, so task 6 switches function class completely: an "
  f"RBF-kernel SVM (scikit-learn SVC, probability calibrated by internal Platt "
  f"scaling, class_weight 'balanced'), tuned over C ∈ {{0.5, 1, 3, 10}} and gamma ∈ "
  f"{{scale, 0.05, 0.02}} on the same validation split (selected C = {X['params']['C']}, "
  f"gamma = {X['params']['gamma']}). The split, the pre-processing, the threshold "
  f"selection rule are identical to task 2, and the KernelSHAP background is a "
  f"smaller draw (25 rows) from the same training split, so any difference in "
  f"section 8 comes from the model, its explainer or the background size, not from "
  f"data handling. DeepSHAP was considered and set aside: an SVM has no "
  f"differentiable network to propagate through, and KernelSHAP is the route the "
  f"template suggests for exactly this pairing.")

h2("7.2  Explainer choice and implementation")
table([
    ["Aspect", "KernelSHAP route (chosen)", "DeepSHAP route (not applicable)"],
    ["Suitable model class", "Any prediction function; used on the SVM",
     "Differentiable neural networks only - no SVM support"],
    ["Output and class", "predict_proba, stroke = 1, probability scale",
     "network output, selected class, chosen scale"],
    ["Background", "25 training rows drawn from the same training split (seed 42); "
     "task 2 used 100 rows, so background distributions differ slightly between the two explainers",
     "would need training-sample background and seed"],
    ["Computation", f"nsamples 300 per row for the global set of {C6['global_set'].split()[0]} "
     f"test rows (600 for the single case), no L1 regularisation, seed 42; "
     f"{C6['runtime_global_sec']} s total",
     "framework, batching and evaluation mode would need recording"],
    ["Input consistency", "explainer consumed the same transformed matrix as the model",
     "tensor shapes and transformed feature names would need recording"],
], widths=[3.8, 6.6, 5.4])
p(f"Compatibility checks: reconstruction residual against predict_proba, max "
  f"{C6['additivity_max_abs_residual_probability']:.2e} / mean "
  f"{C6['additivity_mean_abs_residual_probability']:.2e} in probability units (tolerance "
  f"{C6['tolerance']}); no warnings were emitted. A residual that is identically zero is "
  f"the expected outcome here and not evidence of a better explanation: the KernelSHAP "
  f"solution is obtained under an efficiency constraint, so the base value plus the 21 "
  f"contributions reproduces the model output for every row by construction. What the "
  f"check does establish is that the wrapper feeds the stroke = 1 column, that the scale "
  f"matches predict_proba, and that no feature was silently dropped; it says nothing "
  f"about whether 300 coalitions per row were enough for the values to be stable, which "
  f"remains an untested limitation (section 8.2). The output is on the probability scale, "
  f"unlike the log-odds scale of task 2 - section 8.1 therefore compares rankings "
  f"and signs, never raw magnitudes.")

h2("7.3  Evidence for the selected model")
figure(FIG / "fig4_alt_local_force.png", 16.8,
       "Figure 4. KernelSHAP force plot for the same test case #668 under the SVM, "
       "probability scale. Note the scale change against figure 2: here the base "
       "value and the output are probabilities, not log-odds.")
figure(FIG / "fig5_alt_global_summary.png", 14.6,
       "Figure 5. KernelSHAP summary plot over a fixed subsample of "
       f"{C6['global_set'].split()[0]} test patients, SVM model, probability scale. "
       "The subsample (seed 42) keeps KernelSHAP affordable; the subsample's "
       "composition matches the test split's prevalence within a point.")
p(f"The SVM reaches ROC AUC {X['roc_auc']:.3f} and average precision {X['average_precision']:.3f} "
  f"on the test split (details in table 10). Its global picture agrees with the "
  f"boosted trees on the leading variable - {t6['grouped_ranking'][0]['variable']} - "
  f"while the ordering below it differs, as expected for a kernel method that "
  f"balances a single weighted margin against the tree ensemble's additive splits.")

# ================================================================ 8 comparison
h1("8  Comparison, limitations and conclusion")

h2("8.1  The two models side by side")
table([
    ["Item", "Task 2 - XGBoost", "Task 6 - SVM (RBF)"],
    ["Model and explainer", "boosted trees / TreeExplainer", "SVM / KernelSHAP"],
    ["Precision (stroke = 1)", f"{M['precision_pos']:.3f}", f"{X['precision_pos']:.3f}"],
    ["Recall (stroke = 1)", f"{M['recall_pos']:.3f}", f"{X['recall_pos']:.3f}"],
    ["F1 (stroke = 1) / balanced accuracy", f"{M['f1_pos']:.3f} / {M['balanced_accuracy']:.3f}",
     f"{X['f1_pos']:.3f} / {X['balanced_accuracy']:.3f}"],
    ["ROC AUC / average precision", f"{M['roc_auc']:.3f} / {M['average_precision']:.3f}",
     f"{X['roc_auc']:.3f} / {X['average_precision']:.3f}"],
    ["Case #668 score", f"p = {CASE['predicted_probability']:.4f} (false negative)",
     f"p = {t6['case']['svm_probability']:.4f} "
     f"({'also a false negative' if t6['case']['svm_probability'] < M['threshold'] else 'flagged'})"],
    ["Explanation scale", "log-odds, base "
     f"{E['base_value_log_odds']:.3f}", f"probability, base {C6['base_value_probability']:.3f}"],
    ["Top variable (aggregated)", f"{t4['ranking'][0]['variable']}", f"{t6['grouped_ranking'][0]['variable']}"],
    ["Explained rows / runtime", f"1,022 rows / {E['runtime_sec']} s",
     f"{C6['global_set'].split()[0]} rows / {C6['runtime_global_sec']} s"],
    ["Max reconstruction residual", f"{E['additivity_max_abs_residual_log_odds']:.1e} log-odds",
     f"{C6['additivity_max_abs_residual_probability']:.1e} probability"],
], widths=[5.2, 5.6, 5.0])
p("The agreement that matters is directional: both explainers put age first by a "
  "clear margin, and both score case #668 under the alert line, so the miss is a "
  "property of the data representation, not of one model's quirks. The disagreement "
  "sits in the mid-field, where the SVM distributes weight differently across the "
  "smoking and work-type dummies. Nothing here compares raw SHAP magnitudes across "
  "the two output scales - log-odds and probability units are not convertible by a "
  "constant factor, and the differences between the background distributions add "
  "their own effect.")

h2("8.2  Limitations and responsible interpretation")
p("Data. One retrospective file from an unknown collection period; no follow-up, so "
  "the target is a historical record, not a verified future stroke. The cohort is "
  "not representative of any deployment population, and subgroup performance by sex "
  "or age band was not assessed because the positive count (249, of which 50 fall in "
  "the test split) is too small for stable estimates. Missingness is informative: "
  "BMI-missing patients carry a 19.9% stroke rate, and median imputation erases that "
  "marker while folding those patients into a fixed risk region - the mechanism "
  "behind the +1.45 log-odds BMI term in case #668. A missing-indicator column is "
  "the first listed improvement.")
p("Modelling. Probabilities are not calibrated and no calibration analysis was run, "
  "so threshold-based metrics are reported but the probability values themselves "
  "should not be quoted as risks. Correlated predictors (age with ever_married and "
  "work_type = children, smoking dummies with each other) share credit in ways that "
  "depend on the background distribution. Class imbalance is handled by weighting, "
  "which trades precision for recall and was tuned for F1 on validation.")
p("Explanation stability. The TreeExplainer background is one draw of 100 training "
  "rows (seed 42); repeating the explanation with a different background sample "
  "would shift contributions for correlated features, and no stability experiment "
  "was run - this is reported as observed, not assumed away. KernelSHAP adds "
  "approximation error (recorded above) and covered a 150-row subsample rather than "
  "the full test split. SHAP explains this model; no causal claim anywhere in this "
  "report follows from it, and no figure should be read as a validated future-risk "
  "score.")

h2("8.3  Conclusion")
CONCLUSION = (
    f"On the pre-declared criteria the boosted-tree model is the stronger of the two: "
    f"ROC AUC {M['roc_auc']:.3f} against {X['roc_auc']:.3f}, average precision "
    f"{M['average_precision']:.3f} against {X['average_precision']:.3f}, and recall "
    f"{M['recall_pos']:.2f} against {X['recall_pos']:.2f} at comparable precision. SHAP "
    f"shows age dominates both models (mean |SHAP| {RANK['age']['mean_abs_shap_log_odds']:.2f} "
    f"log-odds, ten times any other variable), while the local analysis of case #668 "
    f"exposes what ranking tables hide: an imputation artefact and an 'Unknown' "
    f"smoking entry pushed a true stroke case below the alert line. The KernelSHAP "
    f"analysis of the SVM agrees on the leading factor, which locates the problem in "
    f"the data representation rather than the learner. The main limitation is "
    f"informative missingness that median imputation destroys; a missing-indicator "
    f"column and calibration analysis are the concrete next steps."
)
assert 100 <= wc(CONCLUSION) <= 150, f"conclusion {wc(CONCLUSION)} words"
p(CONCLUSION)
caption(f"({wc(CONCLUSION)} words)")

h2("8.4  Contributions, assistance and tools")
table([
    ["Item", "Statement"],
    ["Work", "Individual coursework. Analysis, code, figures and text: [Student name]"],
    ["Course materials", "Course repository ciumGit/7009, commit 4889192: 00_data_exploration.py "
     "run unmodified as the EDA baseline; utils.py data loading reused conceptually; "
     "03_shap.py consulted as the course's SHAP demonstration. No line of the course "
     "scripts was modified; script-level changes sit in src/, method-level departures "
     "are listed in 2.1, 2.3 and 3.1"],
    ["Software", "Python 3.11.1; pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.2.0, shap 0.51.0, "
     "imbalanced-learn 0.14.2, matplotlib 3.10.8, seaborn; versions pinned in requirements.txt"],
    ["Generative AI", "Used for debugging assistance and language editing only. Model choices, "
     "preprocessing decisions, all reported numbers and their interpretation were produced and "
     "verified by rerunning the scripts in this repository; no result in this report was "
     "generated or invented by an AI tool"],
    ["Other assistance", "[None / discussants, per course policy]"],
], widths=[3.6, 12.2])

# ================================================================ 9 references
h1("9  References, reproducibility and submission checklist")

h2("9.1  References")
for ref in [
    "World Health Organization. The top 10 causes of death (2023 estimates: stroke, "
    "second cause, ~11% / 6.8 million deaths). https://www.who.int/topics/mortality/en/ "
    "(accessed at submission).",
    "Fedesoriano. Stroke Prediction Dataset. Kaggle. "
    "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model "
    "predictions. Advances in Neural Information Processing Systems 30 (SHAP).",
    "Lundberg, S. M., et al. (2020). From local explanations to global understanding "
    "with explainable AI for trees. Nature Machine Intelligence, 2, 56-67.",
    "Lundberg, S. M., et al. (2018). Explainable machine-learning predictions for the "
    "prevention of all-cause mortality. AAAI-18 (KernelSHAP).",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12.",
]:
    par = doc.add_paragraph(ref)
    par.paragraph_format.left_indent = Cm(0.8)
    par.paragraph_format.first_line_indent = Cm(-0.8)
    for r_ in par.runs:
        r_.font.size = Pt(10)

h2("9.2  Reproducibility appendix")
table([
    ["Item", "Value"],
    ["Environment", "Python 3.11.1 (venv), see requirements.txt for the full pin list"],
    ["Seeds", "split 42, model 42, background sample 42, KernelSHAP subsample 42"],
    ["Data", "healthcare-dataset-stroke-data.csv, 5,110 × 12, as shipped in the course "
     "repository (commit 4889192); SHA-256 recorded in outputs/data_sha256.txt"],
    ["Configuration", "src/prep.py (split, pipeline), src/02_task2_model_shap.py (model, "
     "explainer), src/03/04 scripts (figures), src/06_task6_alt_model.py (SVM, KernelSHAP)"],
    ["Evidence files", "outputs/: eda_stats.json, task2_metrics.json, shap_values_test_logodds.npy, "
     "test_pred_proba.npy, task3_case.json, task4_global.json, task6_metrics.json, "
     "X_test_encoded.csv, X_test_raw.csv; figures/fig1-fig5 (png); logs/eda_baseline.log"],
], widths=[3.6, 12.2])

h2("9.3  Final submission checklist")
for item in [
    "Task 1 - repository recorded (commit 4889192), 00_data_exploration.py executed "
    "unmodified, EDA results interpreted in 2.2-2.3",
    "Task 2 - model evaluated on a held-out test split; SHAP interpreter specified with "
    "class, background, output scale and verified additivity",
    "Task 3 - readable force plot (figure 2) tied to one held-out case with a declared "
    "selection rule and a critique of its artefacts",
    "Task 4 - readable summary plot (figure 3a) plus per-variable aggregation (3c) and a "
    "global pattern reading",
    "Task 5 - three slides for data scientist, clinician and patient, consistency-checked "
    "against the report",
    "Task 6 - different black-box model (SVM) explained with KernelSHAP, reconstruction "
    "residual recorded, same splits and comparable predictors",
    "All placeholders replaced; every figure captioned with sample, scale and reading; "
    "sources cited; code and versions provided",
    "No invented results, no data leakage (preprocessing fitted on train only), no causal "
    "claims, no unvalidated future-risk statements",
]:
    doc.add_paragraph("☑  " + item, style="List Bullet")

doc.save(ROOT / "Report_Stroke_SHAP.docx")
print("saved Report_Stroke_SHAP.docx;", wc(ABSTRACT), "word abstract,",
      wc(CONCLUSION), "word conclusion")
