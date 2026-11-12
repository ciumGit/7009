"""Build the assignment report as a DOCX file from the recorded results.

Run from the repository root:  python analysis/build_report.py
"""
import json
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Inches

OUT = "report"
FIG = "figures"
os.makedirs(OUT, exist_ok=True)

m2 = json.load(open("results/metrics_task2.json"))
gs = json.load(open("results/global_summary.json"))
case = json.load(open("results/case_task3.json"))
m6 = json.load(open("results/metrics_task6.json"))
eda = json.load(open("results/eda_summary.json"))

t2 = m2["test"]
t6 = m6["test"]
grp = {r["pretty"]: r for r in gs["grouped_importance"]}
cg = case["grouped"]
c6 = m6["explainer_detail"]["case_contributions"]

SHA = "644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420"
REPO = "https://github.com/ciumGit/7009/tree/4889192a797956f1c815407c7e6565f42a5fec62/xai-series-master"

doc = Document()

# ---------------------------------------------------------------- page setup
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, attr, Cm(2.5))

# ---------------------------------------------------------------- styles
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.first_line_indent = Pt(24)
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size, before, after in [("Heading 1", 16, 14, 8), ("Heading 2", 14, 12, 6),
                                  ("Heading 3", 12, 10, 4)]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.first_line_indent = Pt(0)
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.line_spacing = 1.3

title_style = doc.styles["Title"]
title_style.font.name = "Times New Roman"
title_style.font.size = Pt(18)
title_style.font.bold = True
title_style.font.color.rgb = RGBColor(0, 0, 0)
title_style.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_style.paragraph_format.space_after = Pt(6)


def H1(text):
    p = doc.add_paragraph(text, style="Heading 1")
    p.paragraph_format.page_break_before = True if text.startswith(("1 ", "2 ", "3 ", "4 ",
                                                                    "5 ", "6 ", "7 ", "8 ",
                                                                    "9 ")) else False
    return p


def H2(text):
    return doc.add_paragraph(text, style="Heading 2")


def P(text, indent=True, size=12, space_after=5, italic=False, align=None):
    p = doc.add_paragraph()
    if not indent:
        p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.italic = italic
    return p


def PB(lead, rest, indent=True):
    """Paragraph opening with a bold lead phrase."""
    p = doc.add_paragraph()
    if not indent:
        p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    r1 = p.add_run(lead)
    r1.bold = True
    r1.font.size = Pt(12)
    r2 = p.add_run(rest)
    r2.font.size = Pt(12)
    return p


def shade(cell, hex_color="E7E6E6"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def no_split(row):
    trPr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    trPr.append(cant)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trPr.append(hdr)


def set_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=10.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(0)
    p.alignment = align
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def table(caption, headers, rows, widths_cm, alignments=None, font_size=10.5):
    cap = doc.add_paragraph()
    cap.paragraph_format.first_line_indent = Pt(0)
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.keep_with_next = True
    r1 = cap.add_run(caption.split(":", 1)[0] + ":")
    r1.bold = True
    r1.font.size = Pt(10.5)
    if ":" in caption:
        r2 = cap.add_run(caption.split(":", 1)[1])
        r2.font.size = Pt(10.5)

    tb = doc.add_table(rows=1, cols=len(headers))
    tb.style = "Table Grid"
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    tb.autofit = False
    hdr = tb.rows[0]
    repeat_header(hdr)
    no_split(hdr)
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        c.width = Cm(widths_cm[i])
        set_cell(c, h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=font_size)
        shade(c)
    for row_data in rows:
        row = tb.add_row()
        no_split(row)
        for i, val in enumerate(row_data):
            c = row.cells[i]
            c.width = Cm(widths_cm[i])
            al = alignments[i] if alignments else (WD_ALIGN_PARAGRAPH.CENTER if i == 0
                                                  else WD_ALIGN_PARAGRAPH.LEFT)
            set_cell(c, str(val), align=al, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tb


def figure(path, caption, width=15.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Pt(0)
    cap.paragraph_format.space_after = Pt(8)
    r1 = cap.add_run(caption.split(":", 1)[0] + ":")
    r1.bold = True
    r1.font.size = Pt(10.5)
    r2 = cap.add_run(caption.split(":", 1)[1])
    r2.font.size = Pt(10.5)


def add_page_number_footer():
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for instr in ("begin", None, "end"):
        if instr is None:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
            run._r.append(el)
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), instr)
            run._r.append(el)


add_page_number_footer()

# ---------------------------------------------------------------- title block
p = doc.add_paragraph(style="Title")
p.add_run("Stroke Classification and SHAP Explanation")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Assignment report")
r.font.size = Pt(12)
r.bold = True

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
p.paragraph_format.space_after = Pt(24)
r = p.add_run("Exploratory data analysis, local and global explanations, audience communication, "
              "and a second black box model")
r.font.size = Pt(12)

for label in ["Student name: [Enter name]", "Student ID: [Enter ID]",
              "Course and programme: [Enter details]", "Lecturer: [Enter name]",
              "Submission date: [Enter date]"]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(label)
    r.font.size = Pt(11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
p.paragraph_format.space_before = Pt(8)
r = p.add_run("Repository: ")
r.font.size = Pt(10)
r2 = p.add_run(REPO)
r2.font.size = Pt(10)
r2.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

# ---------------------------------------------------------------- contents
p = doc.add_paragraph()
p.paragraph_format.page_break_before = True
p.paragraph_format.first_line_indent = Pt(0)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Contents")
r.bold = True
r.font.size = Pt(14)

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(0)
run = p.add_run()
fld_begin = OxmlElement("w:fldChar")
fld_begin.set(qn("w:fldCharType"), "begin")
run._r.append(fld_begin)
instr = OxmlElement("w:instrText")
instr.set(qn("xml:space"), "preserve")
instr.text = ' TOC \\o "1-2" \\h \\z \\u '
run._r.append(instr)
fld_sep = OxmlElement("w:fldChar")
fld_sep.set(qn("w:fldCharType"), "separate")
run._r.append(fld_sep)
txt = OxmlElement("w:t")
txt.text = "Right-click and choose Update Field to build the table of contents."
run._r.append(txt)
fld_end = OxmlElement("w:fldChar")
fld_end.set(qn("w:fldCharType"), "end")
run._r.append(fld_end)

# ---------------------------------------------------------------- abstract
H1("Abstract")
P("This report asks whether a routinely collected clinical table can support a stroke classifier, "
  "and whether SHAP explanations make that classifier usable for three different audiences. The "
  "Kaggle stroke dataset contains 5,110 patients and 249 stroke cases (4.87%). The data were split "
  "60/20/20 with stratification; body mass index was imputed with a training-set median and "
  "categorical variables were one-hot encoded. A random forest of 300 trees, trained on oversampled "
  "training rows, reached recall 0.76, precision 0.16, F1 0.26 and ROC AUC 0.855 on 1,022 held-out "
  "patients at a threshold of 0.56. A TreeExplainer on the class-1 probability reconstructed the "
  "model output to within 0.009. Age carried 60% of mean absolute SHAP, while BMI, marital status "
  "and blood sugar contributed about 8% each. For a true positive with a predicted probability of "
  f"{case['p_stroke']:.2f}, age (+{cg['Age']:.2f}) and heart disease (+{cg['Heart disease']:.2f}) "
  "dominated the force plot. A calibrated RBF SVM explained with Kernel SHAP reached ROC AUC "
  f"{t6['roc_auc']:.3f} and recall {t6['recall']:.2f}; it agreed that age mattered most but "
  "disagreed with the forest on heart disease for the same patient. Weak calibration and the "
  "retrospective, unbalanced source data limit what these explanations can support.")

# ---------------------------------------------------------------- report map
table("Table 1: Report map",
      ["Assignment component", "Report location"],
      [["Task 1 Exploratory data analysis", "Sections 2.1 to 2.3"],
       ["Task 2 Create a SHAP explainer", "Section 3"],
       ["Task 3 Local force plot", "Section 4"],
       ["Task 4 Global summary plot", "Section 5"],
       ["Task 5 Three audience slides", "Section 6"],
       ["Task 6 Deep SHAP or Kernel SHAP", "Sections 7 and 8"]],
      [9.0, 7.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

# ---------------------------------------------------------------- 1
H1("1 Study Context and Dataset")
H2("1.1 Background and objective")
P("Stroke leaves survivors with long-term impairment and accounts for a large share of deaths "
  "worldwide. The assignment brief cites the World Health Organization for two claims: that stroke "
  "is the second leading cause of death globally and that it causes about 11% of all deaths. "
  "Checking those claims dates them to 2019. The Global Burden of Disease 2019 stroke analysis "
  "reports stroke as the second leading cause of death, with 11.6% (95% UI 10.8-12.2) of total "
  "deaths in that year [1]; the WHO top-ten causes fact sheet lists stroke second in 2019 but "
  "third in 2021, when COVID-19 entered the ranking [2]. The figures in the brief therefore "
  "describe 2019 and are not presented here as current.")
P("A classifier built on tabular patient records can rank patients by recorded risk, but a ranking "
  "without an account of what produced it is hard to defend in a clinical setting. Explanations "
  "serve two purposes here: they let an analyst check whether the model relies on plausible "
  "variables, and they give clinicians and patients a way to discuss one prediction in language "
  "they can follow.")
PB("Objective: ", "to evaluate a random forest classifier on held-out patients and to explain both "
  "individual and population-level predictions with SHAP, then to repeat the explanation for a "
  "structurally different model and compare the two accounts.")
PB("Outcome definition: ", "stroke = 1 marks a patient with a recorded history of stroke and "
  "stroke = 0 a patient without one. The dataset carries no follow-up period and no time stamp, so "
  "the model output is described throughout as a probability of a recorded stroke, not as a "
  "validated future risk.")

H2("1.2 Data source and variables")
P("Dataset URL: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset, accessed on "
  "12 November 2026. The file used is healthcare-dataset-stroke-data.csv from the course "
  "repository, with SHA-256 fingerprint "
  f"{SHA[:16]}...{SHA[-8:]}. It holds 5,110 rows and 12 columns. Population, collection period and "
  "provenance beyond the Kaggle description are not stated in the source and are reported here as "
  "unavailable.")
table("Table 2: Variables and analytical roles",
      ["Attribute", "Meaning or values", "Analytical role"],
      [["id", "Unique identifier", "Excluded from predictors"],
       ["gender", "Male, Female, Other", "Categorical predictor"],
       ["age", "Patient age in years (0.08 to 82)", "Numeric predictor"],
       ["hypertension", "0 = no, 1 = yes", "Binary predictor, kept as one column"],
       ["heart_disease", "0 = no, 1 = yes", "Binary predictor, kept as one column"],
       ["ever_married", "No, Yes", "Categorical predictor"],
       ["work_type", "children, Govt_job, Never_worked, Private, Self-employed", "Categorical predictor"],
       ["Residence_type", "Rural, Urban", "Categorical predictor"],
       ["avg_glucose_level", "Average blood glucose in mg/dL", "Numeric predictor"],
       ["bmi", "Body mass index in kg/m2, 201 missing", "Numeric predictor, median imputation"],
       ["smoking_status", "formerly smoked, never smoked, smokes, Unknown", "Categorical predictor"],
       ["stroke", "0 = no recorded stroke, 1 = recorded stroke", "Target variable"]],
      [3.6, 7.4, 5.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
P("Data checks confirmed the column names and category spellings in the file. The brief lists "
  "Govt_jov; the file spells the level Govt_job, and the file spelling is used. Unknown in "
  "smoking_status means the information is unavailable; it is kept as a separate level and never "
  "treated as never smoked.")

# ---------------------------------------------------------------- 2
H1("2 Task 1 Exploratory Data Analysis")
H2("2.1 Code download and execution")
P(f"The course repository was obtained as a ZIP archive of branch commit 4889192 and unpacked "
  f"locally. The exploratory script shipped with the repository is 00_data_exploration.py, which "
  f"loads the CSV through utils.DataLoader, prints the shape and column information, draws one "
  f"histogram per column and then prints the preprocessed frame. It was executed on 12 November "
  f"2026 with python 00_data_exploration.py from the repository root (Python 3.12.11, pandas 2.2.3, "
  f"numpy 1.26.4, matplotlib 3.11.0). The run completed without errors and printed 5,110 rows and "
  f"12 columns. The reproducible rerun writes its panel to figures/fig1_eda_panel.png and its "
  f"summary statistics to results/eda_summary.json; the original script itself displayed its "
  f"histograms interactively.")
P("Two departures from the supplied code were recorded. First, the interactive histograms were "
  "replaced by a self-contained panel (analysis/task1_eda.py) so that Figure 1 and the summary "
  "statistics can be regenerated from one command; the underlying reads and checks are unchanged. "
  "Second, utils.preprocess_data fills missing BMI with 0, which would place 201 patients at an "
  "impossible body mass. That step was not used; the modelling pipeline imputes the training median "
  "instead, and the difference is reported in Section 2.3.")

H2("2.2 Data quality and descriptive results")
rows = [
    ["Rows and columns", f"{eda['n_rows']} rows; {eda['n_cols']} columns",
     "Imported dimensions match the file; id is unique."],
    ["Duplicate records or IDs", f"{eda['duplicate_rows']} duplicate rows; {eda['duplicate_ids']} duplicate IDs",
     "All rows retained; no de-duplication needed."],
    ["Missing values by variable",
     f"bmi {eda['missing']['bmi']} ({eda['missing_pct']['bmi']}%); no missing elsewhere",
     "Median imputation fitted on training rows; smoking Unknown kept as a level (1,544; 30.2%)."],
    ["Stroke class counts",
     f"{eda['stroke_counts']['0']} no stroke; {eda['stroke_counts']['1']} stroke "
     f"({eda['stroke_prevalence']}%)",
     "Strong imbalance; recall and precision reported alongside accuracy."],
    ["Age, glucose and BMI",
     f"age {eda['age_overall']['mean']} ± {eda['age_overall']['sd']} "
     f"(median {eda['age_overall']['median']:.0f}, range {eda['age_overall']['min']:.2f}-"
     f"{eda['age_overall']['max']:.0f}); glucose {eda['avg_glucose_level_overall']['mean']} ± "
     f"{eda['avg_glucose_level_overall']['sd']} (median {eda['avg_glucose_level_overall']['median']:.1f}); "
     f"BMI median {eda['bmi_overall']['median']:.1f} (IQR {eda['bmi_overall']['q1']:.1f}-"
     f"{eda['bmi_overall']['q3']:.1f}, max {eda['bmi_overall']['max']:.1f})",
     f"{eda['glucose_gt_200']} glucose values above 200 mg/dL, {eda['bmi_gt_60']} BMI values above "
     f"60 and {eda['age_under_1']} records under one year kept but flagged."],
    ["Categorical distributions",
     "gender Female 2,994 (58.6%), Male 2,115 (41.4%), Other 1; smoking never 1,892 (37.0%), "
     "Unknown 1,544 (30.2%), formerly 885 (17.3%), smokes 789 (15.4%); work Private 2,925 (57.3%), "
     "Self-employed 819 (16.0%), children 687 (13.4%), Govt_job 657 (12.9%), Never_worked 22 (0.4%); "
     "urban residence 2,596 (50.8%)",
     "Rare levels (Other, Never_worked) are sparse; one-hot encoding with handle_unknown handles "
     "unseen levels at prediction time."],
]
table("Table 3: Data quality checks and actions", ["Check or statistic", "Observed result",
                                                  "Action or interpretation"], rows,
      [3.4, 7.6, 5.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

figure(f"{FIG}/fig1_eda_panel.png",
       "Figure 1: Exploratory overview for the 5,110-patient stroke table. (a) class counts; "
       "(b) age by outcome; (c) average glucose by outcome; (d) BMI distribution with missing count; "
       "(e) smoking status with stroke rate on the right axis; (f) stroke rate by age band with "
       "group sizes. Stroke rate rises from 0.2% below 18 to 16.2% at 65 and over, and the positive "
       "class is 4.87% of the sample.",
       width=16.2)

H2("2.3 Interpretation and preprocessing decisions")
P("Three patterns in Figure 1 shaped the modelling plan. First, age carries most of the visible "
  "risk gradient: stroke rate moves from 0.2% in patients under 18 to 16.2% in those aged 65 and "
  "over, roughly an eightyfold difference (Figure 1f). A model that does not use age well will "
  "therefore lose most of its signal, and any explanation that is not age-led deserves checking.")
P("Second, average glucose is right-skewed with a long upper tail; 434 values exceed 200 mg/dL and "
  "stroke cases are concentrated at the high end (Figure 1c). BMI is closer to a normal "
  "distribution but has 201 missing values and 13 records above 60 (Figure 1d). Missing BMI is "
  "concentrated among patients whose other fields are present, so it is treated as missing at "
  "random and imputed with the training median rather than dropped.")
P("Third, former smokers show a higher stroke rate (7.9%) than current smokers (5.3%) and never "
  "smokers (4.8%), while the Unknown group has the lowest rate at 3.0% (Figure 1e). This ordering "
  "is an association in a cross-sectional sample and not a causal statement; it may reflect age "
  "composition and who was asked about smoking. It does mean that smoking_status should be kept as "
  "four levels, with Unknown distinguished rather than merged.")
P("Preprocessing follows the split, not the other way round. The data were divided 60/20/20 with "
  "stratification on the target and a fixed seed, and every fitted step - median imputation, "
  "one-hot encoding, and random oversampling of the minority class - was fitted on the training "
  "rows only. The validation fold was used to choose hyperparameters and the decision threshold; "
  "the test fold was touched once, for the numbers in Section 3. id and stroke were excluded from "
  "the predictors, leaving ten original variables and 20 columns after encoding.")

# ---------------------------------------------------------------- 3
H1("3 Task 2 Model and SHAP Explainer")
H2("3.1 Model development")
P(f"The Task 2 classifier is a random forest with {m2['best_params']['n_estimators']} trees, "
  f"maximum depth {m2['best_params']['max_depth']} and minimum leaf size "
  f"{m2['best_params']['min_samples_leaf']}. Three configurations were compared on the validation "
  f"fold by positive-class F1 at a 0.5 cut-off and the middle setting won with "
  f"{m2['validation_f1_at_0.5']:.3f}. A forest was chosen because it handles mixed scales without "
  f"further transformation, gives a probability output, and has an exact tree explainer, which "
  f"leaves Kernel SHAP free for the genuinely different model in Task 6.")
P(f"Predictors kept: age, average glucose, BMI, hypertension, heart disease, gender, marital "
  f"status, job type, residence and smoking status. Excluded: id and the target. Preprocessing: "
  f"median imputation for BMI, one-hot encoding for the five categorical variables with "
  f"handle_unknown='ignore', binary variables kept as single 0/1 columns. No scaling was applied, "
  f"as the forest is invariant to monotone transformations.")
P(f"Sizes: {m2['split']['train']:,} training rows ({m2['split']['train_positive']} positive), "
  f"{m2['split']['validation']:,} validation rows ({m2['split']['validation_positive']} positive) "
  f"and {m2['split']['test']:,} test rows ({m2['split']['test_positive']} positive), stratified "
  f"with seed {m2['seed']}. Random oversampling of the minority class produced "
  f"{m2['oversampled_train']:,} training rows; no resampling was applied to validation or test. "
  f"The decision threshold of {m2['threshold']:.3f} was selected on the validation fold by "
  f"maximising positive-class F1 and then fixed for the test fold.")
table("Table 4: Held-out test metrics, random forest",
      ["Test metric", "Result", "Evaluation detail"],
      [["Accuracy", f"{t2['accuracy']:.3f}", f"Threshold {m2['threshold']:.3f}; positive class 4.9%"],
       ["Balanced accuracy", f"{t2['balanced_accuracy']:.3f}", "Mean of sensitivity and specificity"],
       ["Precision, recall and F1 for stroke = 1",
        f"{t2['precision']:.3f} / {t2['recall']:.3f} / {t2['f1']:.3f}",
        f"{t2['confusion']['tp']} of {t2['confusion']['tp'] + t2['confusion']['fn']} stroke cases found"],
       ["ROC AUC and average precision", f"{t2['roc_auc']:.3f} / {t2['average_precision']:.3f}",
        "Scores used, not labels; average precision is the PR-AUC"],
       ["Confusion matrix TN / FP / FN / TP",
        f"{t2['confusion']['tn']} / {t2['confusion']['fp']} / {t2['confusion']['fn']} / "
        f"{t2['confusion']['tp']}",
        f"{t2['confusion']['fp']} false alarms against {t2['confusion']['tp']} correct alerts"]],
      [4.6, 4.4, 7.0], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])
P(f"The recall of {t2['recall']:.2f} is bought with {t2['confusion']['fp']} false positives: at "
  f"this threshold the model flags roughly one in six patients who do not go on to have a recorded "
  f"stroke. Subgroup recall is concentrated in the oldest band - 0.91 above 65 and 0.57 between 50 "
  f"and 64, against 0 of 3 positives between 35 and 49 - which is a consequence of where the "
  f"positive cases sit rather than a modelling choice.")

H2("3.2 Explainer specification")
P(f"Explainer: shap.TreeExplainer (SHAP {m2['explainer']['shap_version']}) with "
  f"model_output='probability' and feature_perturbation='interventional'. The explained quantity "
  f"is the predicted probability of stroke = 1, that is class index 1 in predict_proba; positive "
  f"contributions raise that probability and negative contributions lower it.")
P(f"Background: {m2['explainer']['background_rows']} rows sampled from the training fold after "
  f"preprocessing (seed {m2['seed']}). This is the reference distribution against which "
  f"contributions are measured, so the base value of {m2['explainer']['base_value']:.3f} is the "
  f"average model output on that balanced sample, not the 4.9% prevalence of the raw data. "
  f"Explanation sample: {m2['explainer']['explained_rows']} held-out test rows, sampled with the "
  f"same seed, of which {gs['positive_in_sample']} are true positives.")
P("The pipeline was fitted once on training rows and the transformed matrix is what the forest and "
  "the explainer both see; feature names after transformation are recorded in the results file and "
  "match the explainer input order. Probabilities are not log odds, and contributions are never "
  "added to a probability after conversion from the margin.")

H2("3.3 Verification and interpretation")
resid_max = m2["explainer"]["max_abs_residual"]
resid_mean = m2["explainer"]["mean_abs_residual"]
table("Table 5: Additivity checks for the Task 2 explainer",
      ["Verification item", "Recorded evidence"],
      [["Feature names, order and dimensions",
        f"{m2['n_features']} columns; explainer input shape (300, {m2['n_features']}), matching the "
        f"transformed matrix"],
       ["Correct class and output scale",
        f"Class index 1, probability; base value {m2['explainer']['base_value']:.3f}"],
       ["Base value plus sum of SHAP values",
        f"Case in Section 4: {case['base_value']:.3f} + {case['sum_grouped_shap']:.3f} = "
        f"{case['reconstruction']:.3f} against a model output of {case['p_stroke']:.3f}"],
       ["Model output and reconstruction residual",
        f"Maximum {resid_max:.4f}, mean {resid_mean:.5f} over 300 rows; tolerance set at 0.01"]],
      [5.4, 10.6], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
P(f"The largest reconstruction gap is {resid_max:.3f}, just inside the 0.01 tolerance, while the "
  f"mean gap is {resid_mean:.5f}. Small residuals remain because a forest's probability is an "
  f"average over trees and the interventional expectation is estimated from a finite background; "
  f"the pattern is stable enough for the case-level reading in Section 4 but not exact to the last "
  f"decimal. Positive values mean the feature pushed this patient's probability above the base "
  f"value; negative values pushed it below. None of this establishes that a feature causes stroke: "
  f"SHAP describes the fitted model relative to a reference sample.")

# ---------------------------------------------------------------- 4
H1("4 Task 3 Local Explanation Using a Force Plot")
H2("4.1 Selected held-out case")
P(f"The case is row {case['explained_row']} of the 300 explained held-out rows. The rule was fixed "
  f"before any contribution was inspected: among true positives in the explained sample, take the "
  f"one with the median predicted probability. That avoids the usual failure of choosing the most "
  f"dramatic patient, where the explanation only shows what the model does when it is confident.")
table("Table 6: Case summary", ["Field", "Value"],
      [["Anonymous row reference", f"Row {case['explained_row']} of the explained held-out sample"],
       ["Selection rule", "True positive with median predicted probability"],
       ["Actual class", "stroke = 1"],
       ["Predicted class and probability",
        f"stroke = 1 at {case['p_stroke']:.3f} (threshold {case['threshold']:.3f})"],
       ["Case type", "True positive"],
       ["Explained output and units", f"Probability of stroke = 1"],
       ["Base value in those units", f"{case['base_value']:.3f}"]],
      [5.5, 10.5], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
r = case["raw_record"]
P(f"The patient is a {r['age']:.0f}-year-old woman with recorded heart disease, no recorded "
  f"hypertension, BMI {r['bmi']:.1f}, average glucose {r['avg_glucose_level']:.1f} mg/dL, "
  f"self-employed, living in a rural area and a former smoker. She did have a stroke, so the "
  f"prediction is correct at this threshold.")

H2("4.2 Feature contributions")
figure(f"{FIG}/fig2_force_plot.png",
       f"Figure 2: Local SHAP force plot for the held-out case, class 1, probability scale. The base "
       f"value is {case['base_value']:.2f} on the left and the model output {case['p_stroke']:.2f} "
       f"on the right; red bars push the probability up, blue bars push it down. Labels are drawn "
       f"for the six largest groups after recombining one-hot levels; the remaining four "
       f"contributions sum to +0.002 and are listed in Table 7.",
       width=16.2)
table("Table 7: Grouped contributions for the selected case",
      ["Feature", "Patient value", "SHAP value", "Direction for stroke output"],
      [["Age", f"{r['age']:.0f} years", f"+{cg['Age']:.4f}", "Increases"],
       ["Heart disease", "yes", f"+{cg['Heart disease']:.4f}", "Increases"],
       ["Residence", "rural", f"+{cg['Residence']:.4f}", "Increases"],
       ["Married", "yes", f"+{cg['Married']:.4f}", "Increases"],
       ["Blood sugar", f"{r['avg_glucose_level']:.1f} mg/dL", f"+{cg['Blood sugar']:.4f}", "Increases"],
       ["Job", "self-employed", f"{cg['Job']:.4f}", "Decreases"],
       ["Remaining four features combined",
        f"BMI {r['bmi']:.1f}; smoking former; no hypertension; female",
        f"+{cg['Smoking'] + cg['BMI'] + cg['High blood pressure'] + cg['Gender']:.4f}",
        "Increases, net"]],
      [4.4, 4.6, 3.4, 3.6],
      [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER,
       WD_ALIGN_PARAGRAPH.LEFT])

H2("4.3 Interpretation of this prediction")
P(f"Starting from the base value of {case['base_value']:.2f}, the model output moved to "
  f"{case['p_stroke']:.2f} on the probability scale. Age supplied almost the whole movement "
  f"(+{cg['Age']:.2f}); at 80 the patient sits in the band where the observed stroke rate is "
  f"16.2%, and the forest has learned that gradient. Heart disease added a smaller but clear "
  f"amount (+{cg['Heart disease']:.2f}), and rural residence and marital status added "
  f"+{cg['Residence']:.2f} and +{cg['Married']:.2f}. The only sizeable negative contribution was "
  f"job type, self-employed at {cg['Job']:.3f}; BMI and the absence of hypertension each cost "
  f"about a thousandth of probability, too small to matter for this patient.")
P("The explanation is plausible but not comfortable. Almost all of the decision rests on age, which "
  "means the model's confidence here reflects a demographic prior as much as this patient's "
  "clinical picture; the low glucose reading (66.7 mg/dL, below the sample median) contributes "
  "almost nothing, which is consistent with a model that has learned a monotone age effect more "
  "strongly than a glucose effect. If this case had been a false positive, the same plot would "
  "show what to inspect: an old patient with heart disease and no other supporting findings. This "
  "is a description of a model prediction and not a diagnosis, and nothing in the plot suggests a "
  "treatment decision.")

# ---------------------------------------------------------------- 5
H1("5 Task 4 Global Explanation Using a Summary Plot")
H2("5.1 Population and plotting settings")
P(f"Model and output: the Task 2 random forest, probability of stroke = 1. Explanation sample: "
  f"{gs['n_explained']} held-out rows containing {gs['positive_in_sample']} positives, drawn with "
  f"seed {m2['seed']}; the base value is {gs['base_value']:.3f}. Twelve features are displayed in "
  f"the beeswarm and the full 20-column ranking is kept in the results file.")
P("Feature representation: encoded columns are shown individually in Figure 3, then recombined in "
  "Figure 3b and Table 8. The aggregation rule is applied per patient before any averaging: signed "
  "contributions of all levels of one variable are summed for each case, and the grouped mean "
  "absolute SHAP is then the mean of the absolute grouped sums. Summing first matters, because a "
  "level-by-level mean absolute value would count two opposing effects as two separate "
  "importances.")
figure(f"{FIG}/fig4_summary_beeswarm.png",
       f"Figure 3: Global SHAP summary for the {gs['n_explained']} held-out patients, class 1, "
       f"probability scale. Each point is one patient; the horizontal axis is the change in "
       f"predicted stroke probability, colour is the feature value, and features are ordered by "
       f"mean absolute SHAP. Age spreads widest, with high ages to the right; BMI and glucose show "
       f"narrower positive bands.",
       width=15.0)

H2("5.2 Global importance and patterns")
imp = [grp["Age"], grp["BMI"], grp["Marital status"], grp["Blood sugar"], grp["Job type"]]
table("Table 8: Top five predictors after recombining encoded columns",
      ["Rank", "Feature", "Mean absolute SHAP", "Observed pattern"],
      [["1", "Age", f"{imp[0]['mean_abs_shap']:.3f} ({imp[0]['share_pct']}%)",
        "Monotone: points above 60 sit far to the right, children to the left"],
       ["2", "BMI", f"{imp[1]['mean_abs_shap']:.3f} ({imp[1]['share_pct']}%)",
        "High BMI (>35) pushes the probability up; low BMI (<22) pulls it down"],
       ["3", "Marital status", f"{imp[2]['mean_abs_shap']:.3f} ({imp[2]['share_pct']}%)",
        "Mixed: married patients spread both sides of zero, reflecting age"],
       ["4", "Blood sugar", f"{imp[3]['mean_abs_shap']:.3f} ({imp[3]['share_pct']}%)",
        "Right tail above 150 mg/dL increases the output; the middle is flat"],
       ["5", "Job type", f"{imp[4]['mean_abs_shap']:.3f} ({imp[4]['share_pct']}%)",
        "Self-employed and private workers differ; children sit lower"]],
      [1.6, 3.8, 4.4, 6.2],
      [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER,
       WD_ALIGN_PARAGRAPH.LEFT])
P(f"Age accounts for {imp[0]['share_pct']}% of the total grouped attribution, and the top three "
  f"variables account for {gs['share_top3']}%. That concentration is the single most important "
  f"finding of this section: the model is close to an age rule with smaller corrections from BMI, "
  f"marital status, glucose and job type. Colour in Figure 3 is the feature value, not severity; "
  f"the blue-to-red scale for job type, for example, encodes category levels and must not be read "
  f"as a clinical gradient.")
figure(f"{FIG}/fig4b_grouped_importance.png",
       "Figure 3b (supplementary): Mean absolute SHAP after summing one-hot levels per variable, "
       "with the share of total attribution printed beside each bar. Values are on the probability "
       "scale and are not comparable with raw log-odds magnitudes.",
       width=13.5)

H2("5.3 Link local and global findings")
P(f"The case in Section 4 follows the global pattern. Age is first in both, and the case's "
  f"+{cg['Age']:.2f} sits at the upper end of the age distribution in Figure 3 rather than outside "
  f"it. Heart disease is the interesting exception: it adds +{cg['Heart disease']:.2f} for this "
  f"patient but ranks last globally at {grp['Heart disease']['mean_abs_shap']:.3f}. Two things "
  f"explain the gap. Only a small share of patients have recorded heart disease, so its average "
  f"effect is diluted, and the variable overlaps with age, which lets the forest split the same "
  f"signal between correlated inputs.")
P("Global importance summarises magnitude, not direction, and the beeswarm should be read "
  "accordingly: a feature can be important while pushing some patients up and others down. "
  "Correlation also means attribution is partly arbitrary between variables that carry the same "
  "information; age, marital status and job type are entangled in this dataset, and a different "
  "background sample would shift their shares.")

# ---------------------------------------------------------------- 6
H1("6 Task 5 Plan for Three Audience Slides")
P("Three slides were produced as a separate file, Stroke_SHAP_Three_Slides.pptx, one per audience, "
  "all built on the same model, the same case and the same reported numbers. The plan below states "
  "what each slide carries and why.")
table("Table 9: Slide plan",
      ["Slide and audience", "Title", "Visual", "Message carried"],
      [["1, data scientist", "Model performance and what drives the stroke prediction",
        "Global beeswarm with a metrics strip",
        "Split, threshold and held-out metrics; age at 60% of attribution; TreeExplainer on "
        "probability with a balanced background; known weakness: calibration and one-hot splitting"],
       ["2, doctor", "Reading one patient's prediction in context",
        "Force plot of the same patient in plain labels",
        "Probability 0.69 against a 0.56 cut-off; age and heart disease push up; 12 of 50 strokes "
        "missed overall and 0 of 3 in the 35-49 band; score over-predicts, so rank rather than quote"],
       ["3, patient", "What this computer result means for you",
        "Grouped contribution bars in everyday words",
        "First-person wording; age lifted the score most, heart disease and rural residence added a "
        "little; the score can be wrong; no diagnosis, no forecast, no advice"]],
      [3.2, 4.2, 3.8, 5.4],
      [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
       WD_ALIGN_PARAGRAPH.LEFT])
P("Consistency check: the case, predicted value (0.69), output units (probability), threshold "
  f"({case['threshold']:.2f}) and leading features (age, heart disease, residence) agree across the "
  "three slides and with Sections 4 and 5. Slide 2 keeps the technical feature names behind plain "
  "labels, slide 3 drops the base value and the axis title in favour of 'raised or lowered the "
  "score'. Each slide carries a one-line source note; body text is 13 to 14.5 pt, which stays "
  "readable at presentation size.")

# ---------------------------------------------------------------- 7
H1("7 Task 6 Explain a Different Black Box Model")
H2("7.1 Alternative model and rationale")
P(f"Task 2 model: random forest. Task 6 model: a support vector machine with an RBF kernel "
  f"(C = 0.5, gamma = 'scale'), wrapped in three-fold Platt scaling to obtain probabilities. The "
  f"two models differ in hypothesis class, decision boundary and how they handle feature "
  f"interactions, which is what makes the comparison informative; a second forest with different "
  f"hyperparameters would not qualify.")
P(f"The motivating question was whether the age-dominated account survives a model that does not "
  f"partition the space into axis-aligned splits. An SVM draws a smooth boundary in a transformed "
  f"feature space and its explanations come from Kernel SHAP, an approximation rather than an "
  f"exact decomposition. The same split, the same ten predictors and the same oversampling rule "
  f"were used so that differences reflect the model rather than the data.")

H2("7.2 Choose and document one explainer")
table("Table 10: Explainer record for the chosen route",
      ["Record", "Kernel SHAP route (used)", "Deep SHAP route (not used)"],
      [["Suitable explained model", "Predict function of the calibrated SVM",
        "Would need a differentiable network; no network was trained"],
       ["Output and class", "predict_proba, class index 1, probability",
        "Not applicable"],
       ["Background", "shap.kmeans with 20 clusters from the training fold", "Not applicable"],
       ["Computation settings", "nsamples 300 for the global sample, 800 for the local case; "
        "l1_reg 'num_features(10)'", "Not applicable"],
       ["Input consistency", "Same transformed matrix and column order as Task 2",
        "Not applicable"]],
      [3.8, 7.4, 4.8],
      [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
P(f"Implementation: shap.KernelExplainer over the calibrated predict_proba, background of 20 "
  f"k-means centroids computed from the training fold, {m6['explainer_detail']['nsamples_global']} "
  f"coalition samples for the 60-row global sample and "
  f"{m6['explainer_detail']['nsamples_local']} for the single case. Kernel runtime was "
  f"{m6['explainer_detail']['kernel_seconds_global']:.0f} s for the global sample and "
  f"{m6['explainer_detail']['kernel_seconds_local']:.1f} s for the case, against roughly 14 s for "
  f"the whole Task 2 pipeline including its exact explanations.")
P(f"Checks: output shape (60, 20), class index 1, base value "
  f"{m6['explainer_detail']['base_value']:.3f}; the maximum additivity gap over the sample is "
  f"{m6['explainer_detail']['max_abs_residual']:.3f}, so base plus contributions reproduces the "
  f"model output as closely as the algorithm guarantees. Kernel SHAP is exact only in the limit of "
  f"all coalitions; with 300 samples the estimate carries sampling error, and repeated runs shift "
  f"individual values in the third decimal, which is why the numbers below are tied to one "
  f"recorded run rather than quoted as constants.")
P("One practical note: the calibrated SVM's probabilities are compressed relative to the forest's. "
  "Platt scaling maps scores onto the observed frequency in each calibration fold, so the same "
  "patient receives a lower probability from the SVM even though both models rank her above their "
  "thresholds. Comparisons of SHAP magnitudes across these two models are therefore comparisons of "
  "two different probability scales, not of explanatory strength.")

H2("7.3 Required evidence for the chosen model")
table("Table 11: Held-out test metrics, calibrated RBF SVM",
      ["Test metric", "Result", "Evaluation detail"],
      [["Accuracy", f"{t6['accuracy']:.3f}", f"Threshold {m6['threshold']:.3f}"],
       ["Balanced accuracy", f"{t6['balanced_accuracy']:.3f}", "Sensitivity 0.32, specificity 0.91"],
       ["Precision, recall and F1 for stroke = 1",
        f"{t6['precision']:.3f} / {t6['recall']:.3f} / {t6['f1']:.3f}",
        f"{t6['confusion']['tp']} of {t6['confusion']['tp'] + t6['confusion']['fn']} stroke cases found"],
       ["ROC AUC and average precision", f"{t6['roc_auc']:.3f} / {t6['average_precision']:.3f}",
        "Probability output from Platt scaling"],
       ["Brier score", f"{t6['brier']:.3f}", "Squared error of the probabilities"],
       ["Confusion matrix TN / FP / FN / TP",
        f"{t6['confusion']['tn']} / {t6['confusion']['fp']} / {t6['confusion']['fn']} / "
        f"{t6['confusion']['tp']}", "Fewer false alarms, more missed strokes than Task 2"]],
      [4.6, 4.4, 7.0],
      [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])
figure(f"{FIG}/fig5_kernel_local.png",
       f"Figure 4: Local explanation of the calibrated SVM for the same patient as Section 4, "
       f"probability scale. Model output {m6['explainer_detail']['case_fx']:.2f}, base "
       f"{m6['explainer_detail']['base_value']:.2f}; age is the largest contributor, job type and "
       f"heart disease pull the score down. Values are Kernel SHAP estimates from one recorded run.",
       width=15.0)
figure(f"{FIG}/fig5_kernel_global.png",
       "Figure 5: Global mean absolute SHAP for the calibrated SVM over 60 held-out patients, "
       "grouped by original variable. Age dominates at 0.215; smoking, BMI and job type follow at "
       "0.033, 0.031 and 0.023. The ordering differs from the forest, where smoking ranked sixth.",
       width=14.5)
P(f"Agreement and disagreement. Both models put age first and give it a similar share of "
  f"attribution, and both push this patient above their thresholds mainly because she is 80. They "
  f"part company on heart disease: the forest adds +{cg['Heart disease']:.2f}, the SVM subtracts "
  f"{c6['Heart disease']:.3f}, and they also disagree in magnitude about job type "
  f"({cg['Job']:.2f} against {c6['Job type']:.2f}). The SVM's global ranking gives more weight to "
  f"smoking than the forest does, which is consistent with a smooth kernel boundary using "
  f"categorical contrasts that the forest handles through many separate splits.")
P("These differences are not a tie to be broken by choosing the more agreeable explanation. The "
  "two models fit different functions; their attributions inherit those differences, and the SVM's "
  "carry an extra layer of sampling approximation. What can be said is narrower and more useful: "
  "age is the one conclusion both models support for this patient, and any statement about heart "
  "disease at the individual level depends on which model is being discussed.")

# ---------------------------------------------------------------- 8
H1("8 Comparison Discussion and Conclusion")
H2("8.1 Compare the two models")
table("Table 12: Comparison of the two models",
      ["Comparison item", "Task 2 random forest", "Task 6 calibrated SVM"],
      [["Model and explainer", "300-tree forest, TreeExplainer (interventional)",
        "RBF SVM, Platt scaling, KernelExplainer"],
       ["Positive class recall and precision", f"{t2['recall']:.3f} / {t2['precision']:.3f}",
        f"{t6['recall']:.3f} / {t6['precision']:.3f}"],
       ["F1, balanced accuracy, ROC AUC",
        f"{t2['f1']:.3f} / {t2['balanced_accuracy']:.3f} / {t2['roc_auc']:.3f}",
        f"{t6['f1']:.3f} / {t6['balanced_accuracy']:.3f} / {t6['roc_auc']:.3f}"],
       ["Average precision and prevalence",
        f"{t2['average_precision']:.3f} / {t2['prevalence']:.3f}",
        f"{t6['average_precision']:.3f} / {t2['prevalence']:.3f}"],
       ["Case score and explained scale",
        f"{case['p_stroke']:.3f}, probability", f"{m6['explainer_detail']['case_fx']:.3f}, probability"],
       ["Top three global features", "Age, BMI, marital status", "Age, smoking, BMI"],
       ["Explanation sample and runtime", "300 rows, exact tree values, no sampling",
        "60 rows, 91 s (Kernel SHAP)"],
       ["Reconstruction residual", f"max {resid_max:.3f}", "max 0.000 (by construction)"]],
      [4.6, 5.7, 5.7],
      [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
P(f"On the forest's own terms it is the stronger model: it finds far more of the stroke cases "
  f"({t2['recall']:.2f} against {t6['recall']:.2f} recall), scores higher on F1, average precision "
  f"and ROC AUC, and costs a fraction of the explanation time. The SVM's advantage is fewer false "
  f"alarms ({t6['confusion']['fp']} against {t2['confusion']['fp']}), which follows from a much "
  f"higher threshold ({m6['threshold']:.2f} against {m2['threshold']:.3f}); that is a threshold "
  f"choice as much as a modelling gain.")
P(f"Neither model produces usable probabilities. The forest's reliability is poor at the top end - "
  f"a mean predicted probability of 0.69 against an observed frequency of 0.20 in the highest "
  f"quantile - and its Brier score is {t2['brier']:.3f}. The SVM, despite Platt scaling, is worse "
  f"on Brier at {t6['brier']:.3f}. Calibration was assessed but not corrected; any sentence that "
  f"quotes a risk percentage from either model would overstate what was measured.")
P("The explanation comparison adds a second lesson. Agreement on the leading feature is not "
  "evidence that the models agree on the patient; the sign of heart disease differs, and the sign "
  "of a SHAP value is the part a clinician would actually use. Where two models disagree on "
  "direction, the honest conclusion is that the dataset does not pin down that effect, not that "
  "one explanation is wrong.")

H2("8.2 Limitations and responsible interpretation")
P(f"Four limits bound this work. The dataset is retrospective and undocumented beyond its Kaggle "
  f"page: no collection period, no recruitment rule and no external validation. The outcome is a "
  f"recorded history of stroke without a follow-up window, so the model cannot be read as a "
  f"prospective risk tool. Class imbalance is severe (4.87%), and the positive cases are "
  f"concentrated in the oldest band, which makes subgroup estimates unstable - 3 positives in the "
  f"35-49 band is not enough to conclude anything about model behaviour there.")
P(f"Missing data and encoding choices also shape the explanations. BMI is missing for 201 patients "
  f"and imputed at the median, which pushes those patients towards the middle of the BMI "
  f"distribution and mutes any real BMI effect. Smoking Unknown is a large group (30.2%) whose "
  f"meaning is unknown rather than negative. One-hot levels split attribution across columns, and "
  f"the grouped figures depend on the aggregation rule stated in Section 5.1; a different rule "
  f"would produce a different ranking even with identical SHAP values.")
P("Explanation stability was tested rather than assumed. Recomputing the forest explanations "
  "with a second 50-row background (seed 777, all other settings unchanged) reproduced the same "
  "five leading groups with one swap: age stayed first, marital status and BMI exchanged second "
  "and third, and no variable entered or left the top five. The local picture was less stable. For "
  "the Section 4 patient, age moved from +0.304 to +0.295 and heart disease from +0.044 to +0.043, "
  "but job type changed sign (-0.008 to +0.015), and the base value moved from 0.300 to 0.288 while "
  "the model output stayed at 0.694. Small contributions therefore carry little evidential weight; "
  "only the dominant ones survived the change of reference distribution (results/stability.json).")
P("Correlated predictors (age, marital status, job type) also share attribution, and the balanced "
  "background raises the base value well above the population prevalence. Both push readers towards "
  "treating SHAP values as effect sizes. They are not: a value of +0.30 says the model moved this "
  "patient 0.30 above the reference average, not that age caused 30 percentage points of risk.")

H2("8.3 Conclusion")
P(f"The random forest performed better on the criteria that matter for a screening-style task: "
  f"recall {t2['recall']:.2f}, F1 {t2['f1']:.2f} and ROC AUC {t2['roc_auc']:.3f} against "
  f"{t6['recall']:.2f}, {t6['f1']:.2f} and {t6['roc_auc']:.3f} for the calibrated SVM. SHAP showed "
  f"that both models are largely age models: age carries 60% of grouped attribution in the forest "
  f"and dominates the SVM sample as well, while BMI, glucose and marital status add smaller "
  f"corrections. The same patient's explanation changed when the model changed, most visibly in "
  f"the sign of heart disease, which is the clearest argument for reporting explanations alongside "
  f"model identity rather than as properties of the data. Poor calibration and the absence of any "
  f"prospective validation remain the binding constraints on interpretation.")

H2("8.4 Work contributions")
P("Individual work: study design, data checks, modelling, SHAP analysis, figures, slides and "
  "report were produced by the author of this submission. Tools used: Python 3.12 with pandas, "
  "scikit-learn, imbalanced-learn, SHAP and matplotlib; format and language checking were assisted "
  "by an AI writing tool. [Adjust this statement to match the course policy on disclosed tools.]")

# ---------------------------------------------------------------- 9
H1("9 References and Submission Checklist")
H2("9.1 References")
refs = [
    "[1] Feigin, V. L. et al. (2021) Global, regional, and national burden of stroke and its risk "
    "factors, 1990-2019: a systematic analysis for the Global Burden of Disease Study 2019. Lancet "
    "Neurology 20(10), 795-820.",
    "[2] World Health Organization (2024) The top 10 causes of death, fact sheet, updated 7 August "
    "2024. https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death.",
    "[3] fedesoriano. Stroke Prediction Dataset. Kaggle, https://www.kaggle.com/datasets/"
    "fedesoriano/stroke-prediction-dataset (accessed 12 November 2026).",
    "[4] Lundberg, S. M. and Lee, S.-I. (2017) A unified approach to interpreting model predictions. "
    "Advances in Neural Information Processing Systems 30, 4765-4774.",
    "[5] Lundberg, S. M. et al. (2020) From local explanations to global understanding with explainable "
    "AI for trees. Nature Machine Intelligence 2, 56-67.",
    "[6] Molnar, C. (2022) Interpretable Machine Learning, 2nd edition, "
    "https://christophm.github.io/interpretable-ml-book/.",
    "[7] scikit-learn developers. scikit-learn user guide: random forests, SVMs and calibration, "
    "https://scikit-learn.org/stable/user_guide.html.",
    "[8] SHAP developers. SHAP documentation: TreeExplainer and KernelExplainer, "
    "https://shap.readthedocs.io/.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(21)
    p.paragraph_format.first_line_indent = Pt(-21)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(ref)
    r.font.size = Pt(10.5)

H2("9.2 Reproducibility appendix")
P(f"Environment: Python 3.12.11 on Linux; pandas 2.2.3, numpy 1.26.4, scikit-learn 1.9.1, SHAP "
  f"0.45.1, imbalanced-learn 0.14.2, matplotlib 3.11.0. Seeds: 2021 for the split, resampling, "
  f"background sampling and explanation sampling. Data: healthcare-dataset-stroke-data.csv, "
  f"SHA-256 {SHA}.")
P("Configuration: stability check used a second 50-row background (seed 777) on the same 300 "
  "rows. Stratified 60/20/20 split; median imputation and one-hot encoding fitted on "
  "training rows; random oversampling to 5,834 rows; random forest of 300 trees, depth 12, leaf "
  "size 20; threshold 0.558; interventional TreeExplainer on probability with 50 background rows "
  "and 300 explained rows; RBF SVM with C = 0.5 and three-fold Platt scaling; KernelExplainer with "
  "20 k-means background clusters, 300 and 800 samples.")
P("Evidence files: results/eda_summary.json, results/metrics_task2.json, results/test_predictions.csv, "
  "results/shap_values_class1.csv, results/global_importance_grouped.csv, results/case_task3.json, "
  "results/case_contributions.csv, results/metrics_task6.json, results/task6_shap_values.csv, "
  "results/stability.json; "
  "figures/fig1_eda_panel.png to fig5_kernel_global.png; models/task2_pipeline.joblib and "
  "models/task6_svm.joblib (calibrated SVM); "
  "slides/Stroke_SHAP_Three_Slides.pptx; analysis/*.py; requirements.txt.")

H2("9.3 Final submission check")
checks = [
    ["Task 1 repository recorded and 00_data_exploration.py executed; EDA results interpreted",
     "Done", "Sections 2.1 to 2.3, Table 3, Figure 1"],
    ["Task 2 model evaluated and SHAP explainer specified with class, background and output scale",
     "Done", "Section 3, Tables 4 and 5"],
    ["Task 3 readable force plot included and linked to one held-out case",
     "Done", "Section 4, Table 6, Figure 2"],
    ["Task 4 readable summary plot included and global patterns explained",
     "Done", "Section 5, Table 8, Figures 3 and 3b"],
    ["Task 5 three slides prepared for the data scientist, doctor and patient audiences",
     "Done", "Section 6, Table 9, Stroke_SHAP_Three_Slides.pptx"],
    ["Task 6 different black box model explained using Deep SHAP or Kernel SHAP",
     "Done", "Section 7, Tables 10 and 11, Figures 4 and 5"],
    ["All placeholders replaced; figures captioned; sources cited; code and versions supplied",
     "Partly", "Name, ID, course and date placeholders remain; all other items done"],
    ["No invented results, data leakage, unsupported causal claims or unvalidated future risk claims",
     "Done", "Sections 4.3, 8.2 and 8.3 state the boundaries"],
]
table("Table 13: Final submission checklist", ["Check", "Status", "Where it is evidenced"], checks,
      [8.0, 2.2, 5.8],
      [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT])

doc.save(f"{OUT}/Stroke_SHAP_Report.docx")
print("saved", f"{OUT}/Stroke_SHAP_Report.docx")
