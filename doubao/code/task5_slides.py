"""Task 5: three slides for three audiences, built from the verified results.

Run from the repository root:  python analysis/task5_slides.py
"""
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.util import Inches, Pt
from lxml import etree

OUT = "slides"
os.makedirs(OUT, exist_ok=True)
FIG = "figures"

NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xA6, 0xA6, 0xA6)
ACCENT = RGBColor(0xC0, 0x50, 0x4D)
BG = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

m2 = json.load(open("results/metrics_task2.json"))
m6 = json.load(open("results/metrics_task6.json"))
case = json.load(open("results/case_task3.json"))


def new_slide():
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.22), prs.slide_height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    return s


def textbox(slide, x, y, w, h, kicker, title, title_size=26):
    if kicker:
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
        tfk = tb.text_frame
        tfk.word_wrap = True
        tfk.auto_size = MSO_AUTO_SIZE.NONE
        p = tfk.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = kicker.upper()
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = ACCENT
        r.font.name = "Arial"
    tb = slide.shapes.add_textbox(Inches(x), Inches(y + 0.28), Inches(w), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(title_size)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r.font.name = "Arial"


def bullets(slide, x, y, w, h, items, size=14.5, space=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for k, (lead, rest) in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.level = 0
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space)
        p.line_spacing = 1.12
        # real bullet
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Pt(16)))
                )
        pPr.set("indent", str(int(-Pt(16))))
        bu = etree.SubElement(pPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar")
        bu.set("char", "•")
        if lead:
            r1 = p.add_run()
            r1.text = lead
            r1.font.bold = True
            r1.font.size = Pt(size)
            r1.font.color.rgb = NAVY
            r1.font.name = "Arial"
        r2 = p.add_run()
        r2.text = rest
        r2.font.size = Pt(size)
        r2.font.color.rgb = RGBColor(0x26, 0x26, 0x26)
        r2.font.name = "Arial"


def footer(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(6.95), Inches(12.2), Inches(0.35))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(9)
    r.font.color.rgb = LIGHT
    r.font.name = "Arial"


def trim(path):
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bgc = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bgc).getbbox()
    if not box:
        return path
    pad = 14
    box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
           min(box[2] + pad, im.width), min(box[3] + pad, im.height))
    out = path.replace(".png", "_trim.png")
    im.crop(box).save(out)
    return out


def picture_fit(slide, path, x, y, max_w, max_h):
    from PIL import Image
    w, h = Image.open(path).size
    ar = w / h
    width = max_w
    height = width / ar
    if height > max_h:
        height = max_h
        width = height * ar
    left = x + (max_w - width) / 2
    return slide.shapes.add_picture(path, Inches(left), Inches(y), Inches(width), Inches(height))


# ---------------- Slide 1: data scientist ----------------
s1 = new_slide()
textbox(s1, 0.62, 0.35, 12.4, 1.2, "Slide 1 · for a data scientist",
        "Model performance and what drives the stroke prediction")
bullets(s1, 0.62, 1.55, 5.9, 4.9, [
    ("Model: ", "random forest, 300 trees, depth 12, leaf size 20; 60/20/20 stratified split "
                "(3,066 train / 1,022 validation / 1,022 test)."),
    ("Held-out test: ", "recall 0.76, precision 0.16, F1 0.26, ROC AUC 0.855, average precision "
                        "0.26 at 4.9% prevalence."),
    ("Operating point: ", "threshold 0.56, chosen on validation by positive-class F1; oversampling "
                          "used on training rows only."),
    ("Explainer: ", "TreeExplainer on class-1 probability; base value 0.30 from a 50-row balanced "
                    "background; reconstruction residual ≤ 0.009."),
    ("Global picture: ", "age carries 60% of mean |SHAP|; BMI, marital status and blood sugar about "
                         "8% each; all other variables below 6%."),
    ("Known weaknesses: ", "probabilities over-predict in the top decile, one-hot levels split "
                           "attribution, single seed with no repeated cross-validation."),
], size=13.5, space=7)
picture_fit(s1, f"{FIG}/fig4_summary_beeswarm.png", 6.75, 1.45, 6.3, 4.8)
footer(s1, "Kaggle stroke dataset, 5,110 patients. Random forest + SHAP 0.45.1 TreeExplainer; "
           "300 held-out patients explained.")

# ---------------- Slide 2: doctor ----------------
s2 = new_slide()
textbox(s2, 0.62, 0.35, 12.4, 1.2, "Slide 2 · for a doctor",
        "Reading one patient's prediction in context")
picture_fit(s2, trim(f"{FIG}/fig2_force_plot.png"), 0.62, 1.6, 12.1, 2.4)
bullets(s2, 0.62, 4.1, 5.9, 2.5, [
    ("Case: ", "woman, 80, recorded heart disease, BMI 21.7, glucose 67 mg/dL, former smoker. "
               "Model probability 0.69 (cut-off 0.56); she did have a stroke."),
    ("Drivers: ", "age +0.30 and heart disease +0.04 push the score up; rural residence +0.03; "
                  "job and BMI pull it down slightly."),
], size=13, space=6)
bullets(s2, 6.85, 4.1, 6.0, 2.5, [
    ("Missed cases: ", "12 of 50 strokes in the test fold were missed; recall is 0.91 above 65 "
                       "but 0 of 3 between 35 and 49."),
    ("Calibration: ", "predicted 0.69 against 0.20 observed in the top decile, so use the score "
                      "to rank, not to quote a risk percentage."),
    ("Boundary: ", "SHAP explains the model, not the disease. Decisions still rest on history, "
                   "imaging and examination."),
], size=13, space=6)
footer(s2, "Same model and same anonymised case as Section 4 (true positive, row 268 of the 300"
           " explained held-out rows).")

# ---------------- Slide 3: patient ----------------
s3 = new_slide()
textbox(s3, 0.62, 0.35, 12.4, 1.2, "Slide 3 · for a patient",
        "What this computer result means for you")
picture_fit(s3, f"{FIG}/fig3_patient_view.png", 0.55, 1.55, 7.4, 4.6)
bullets(s3, 8.25, 1.55, 4.7, 5.0, [
    ("", "The program read the details already in my record."),
    ("", "My age (80) lifted the score most. Having heart disease added a little, and so did "
         "living in a rural area."),
    ("", "My body mass index and my job counted slightly against it."),
    ("", "The score came out at 0.69 out of 1, above the cut-off this study used."),
    ("", "It is not a diagnosis and not a forecast. The program can be wrong: in this test it "
         "missed about one in four strokes."),
    ("", "A doctor would read the result together with my history and tests."),
], size=13.5, space=8)
footer(s3, "Plain-language view of the same case; contributions grouped by original variable, "
           "probability scale, base value 0.30.")

prs.save(f"{OUT}/Stroke_SHAP_Three_Slides.pptx")
print("saved", f"{OUT}/Stroke_SHAP_Three_Slides.pptx", len(prs.slides.__iter__.__self__._sldIdLst))
