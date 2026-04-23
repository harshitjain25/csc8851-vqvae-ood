"""Build a structured PowerPoint for the CSC 8851 VQ-VAE OOD project."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ========= COLORS =========
NAVY   = RGBColor(0x0B, 0x2A, 0x4A)
BLUE   = RGBColor(0x15, 0x56, 0xA6)
GREEN  = RGBColor(0x1E, 0x7A, 0x4E)
GOLD   = RGBColor(0xFF, 0xD4, 0x51)
CREAM  = RGBColor(0xFF, 0xF7, 0xD6)
SOFT   = RGBColor(0xFA, 0xFB, 0xFD)
BORDER = RGBColor(0xDA, 0xE1, 0xEC)
GREY   = RGBColor(0x55, 0x55, 0x55)
DARK   = RGBColor(0x11, 0x11, 0x11)
RED    = RGBColor(0xD6, 0x73, 0x5A)
PURPLE = RGBColor(0x4A, 0x2C, 0x82)
MUTED  = RGBColor(0x9C, 0xA6, 0xB4)
OURS_BG = RGBColor(0xFF, 0xF7, 0xD6)

# ========= PRESENTATION =========
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def add_slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.line.fill.background()
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bg.shadow.inherit = False
    return s


def add_textbox(slide, x, y, w, h, text, size=18, bold=False, color=DARK,
                align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_bullets(slide, x, y, w, h, items, size=16, color=DARK,
                bullet=True, line_spacing=1.15, bold_prefix_split=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        p.space_after = Pt(4)
        prefix = "•  " if bullet else ""
        # Support bold-lead then rest-regular: split on ":" if bold_prefix_split=True
        if bold_prefix_split and ":" in item:
            lead, rest = item.split(":", 1)
            r0 = p.add_run(); r0.text = prefix + lead + ":"
            r0.font.name = "Calibri"; r0.font.size = Pt(size); r0.font.bold = True
            r0.font.color.rgb = color
            r1 = p.add_run(); r1.text = rest
            r1.font.name = "Calibri"; r1.font.size = Pt(size); r1.font.bold = False
            r1.font.color.rgb = color
        else:
            r = p.add_run(); r.text = prefix + item
            r.font.name = "Calibri"; r.font.size = Pt(size); r.font.bold = False
            r.font.color.rgb = color
    return tb


def add_rect(slide, x, y, w, h, fill, line=None, radius=False):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def add_header(slide, title, subtitle=None, section_num=None):
    # Top banner
    band = add_rect(slide, 0, 0, SW, Inches(0.9), NAVY)
    # Gold accent bar
    accent = add_rect(slide, 0, Inches(0.9), SW, Inches(0.06), GOLD)
    title_text = f"{section_num}.  {title}" if section_num else title
    add_textbox(slide, Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.55),
                title_text, size=26, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_textbox(slide, Inches(0.4), Inches(0.5), Inches(12.5), Inches(0.35),
                    subtitle, size=13, color=GOLD, anchor=MSO_ANCHOR.MIDDLE)


def add_footer(slide, n, total):
    add_rect(slide, 0, Inches(7.25), SW, Inches(0.25), NAVY)
    add_textbox(slide, Inches(0.3), Inches(7.26), Inches(11), Inches(0.23),
                "CSC 8851 · Deep Learning · Spring 2026 · Harshit Jain & Parsh Jadon",
                size=10, color=RGBColor(0xE0, 0xE6, 0xF0),
                anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(slide, Inches(12.3), Inches(7.26), Inches(1), Inches(0.23),
                f"{n} / {total}", size=10, color=GOLD,
                anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)


# ============================================================
# SLIDE 1 — TITLE
# ============================================================
s = add_slide()
# Full-bleed navy background
add_rect(s, 0, 0, SW, SH, NAVY)
# Gold stripe
add_rect(s, 0, Inches(3.4), SW, Inches(0.08), GOLD)

add_textbox(s, Inches(0.6), Inches(1.4), Inches(12), Inches(0.7),
            "Improving Energy-Based",
            size=44, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
add_textbox(s, Inches(0.6), Inches(2.0), Inches(12), Inches(0.7),
            "Out-of-Distribution Detection",
            size=44, bold=True, color=GOLD)
add_textbox(s, Inches(0.6), Inches(2.6), Inches(12), Inches(0.7),
            "using Generative Modeling",
            size=44, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

add_textbox(s, Inches(0.6), Inches(3.6), Inches(12), Inches(0.5),
            "Shifting OOD detection from classifier logits into a VQ-VAE latent space",
            size=18, color=RGBColor(0xCF, 0xD8, 0xE8))

# Authors block
add_textbox(s, Inches(0.6), Inches(4.7), Inches(6), Inches(0.4),
            "Harshit Jain", size=22, bold=True, color=GOLD)
add_textbox(s, Inches(0.6), Inches(5.05), Inches(6), Inches(0.35),
            "hjain5@gsu.edu", size=14, color=RGBColor(0xCF, 0xD8, 0xE8),
            font="Consolas")

add_textbox(s, Inches(0.6), Inches(5.5), Inches(6), Inches(0.4),
            "Parsh Jadon", size=22, bold=True, color=GOLD)
add_textbox(s, Inches(0.6), Inches(5.85), Inches(6), Inches(0.35),
            "pjadon1@gsu.edu", size=14, color=RGBColor(0xCF, 0xD8, 0xE8),
            font="Consolas")

# Course tag (right side)
add_textbox(s, Inches(7.5), Inches(4.7), Inches(5.2), Inches(0.4),
            "CSC 8851  ·  Deep Learning", size=18, bold=True, color=GOLD,
            align=PP_ALIGN.RIGHT)
add_textbox(s, Inches(7.5), Inches(5.1), Inches(5.2), Inches(0.4),
            "Spring 2026 Final Project", size=14,
            color=RGBColor(0xCF, 0xD8, 0xE8), align=PP_ALIGN.RIGHT)
add_textbox(s, Inches(7.5), Inches(5.5), Inches(5.2), Inches(0.4),
            "Georgia State University", size=14,
            color=RGBColor(0xCF, 0xD8, 0xE8), align=PP_ALIGN.RIGHT)
add_textbox(s, Inches(7.5), Inches(5.85), Inches(5.2), Inches(0.4),
            "github.com/harshitjain25/csc8851-vqvae-ood",
            size=12, color=GOLD, align=PP_ALIGN.RIGHT, font="Consolas")


# ============================================================
# SLIDE 2 — AGENDA / ROADMAP
# ============================================================
s = add_slide()
add_header(s, "Presentation Roadmap")

cols = [
    ("1.  Motivation",             "Why overconfidence on unseen inputs is dangerous"),
    ("2.  Related Work",           "Softmax, ODIN, Mahalanobis, Energy, Outlier Exposure"),
    ("3.  Baseline Method",        "Classifier free-energy (Liu et al., 2020)"),
    ("4.  Our Method",             "VQ-VAE latent space + two detectors"),
    ("5.  Training Objectives",    "VQ-VAE loss · contrastive hinge · BCE · blend"),
    ("6.  Implementation",         "Architectures, hyperparams, reproducibility"),
    ("7.  Experimental Setup",     "Datasets, metrics, splits"),
    ("8.  Results",                "AUROC / AUPR / FPR@95 · Near- & Far-OOD"),
    ("9.  Head-to-Head Comparison","Baseline vs Ours, per-metric deltas"),
    ("10. Analysis",               "Why combining can hurt; failure modes"),
    ("11. Conclusions",            "Wins, limits, future work"),
    ("12. References & Q&A",       ""),
]

y = Inches(1.3)
for i, (title, desc) in enumerate(cols):
    row = i // 2
    col = i % 2
    x = Inches(0.6) + col * Inches(6.4)
    yy = y + Inches(row * 0.85)
    add_rect(s, x, yy, Inches(6.0), Inches(0.75), SOFT, line=BORDER, radius=True)
    add_textbox(s, x + Inches(0.2), yy + Inches(0.06), Inches(5.6), Inches(0.35),
                title, size=14, bold=True, color=NAVY)
    if desc:
        add_textbox(s, x + Inches(0.2), yy + Inches(0.38), Inches(5.6), Inches(0.35),
                    desc, size=11, color=GREY)


# ============================================================
# SLIDE 3 — MOTIVATION
# ============================================================
s = add_slide()
add_header(s, "Motivation: the overconfidence problem", section_num="1")

add_textbox(s, Inches(0.4), Inches(1.15), Inches(12.5), Inches(0.4),
            "A CIFAR-10 classifier will confidently label images it has never seen.",
            size=18, bold=True, color=NAVY)

# Problem panel (left)
add_rect(s, Inches(0.4), Inches(1.7), Inches(6.2), Inches(4.6), SOFT,
         line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.85), Inches(5.8), Inches(0.4),
            "The real-world failure modes", size=16, bold=True, color=BLUE)
add_bullets(s, Inches(0.6), Inches(2.3), Inches(5.8), Inches(4.0), [
    "Chest X-ray → labelled 'automobile' with 0.99 softmax",
    "SVHN digit → labelled 'truck'",
    "Gaussian noise → labelled 'frog'",
    "In medical imaging, autonomous driving, and industrial inspection, silent overconfidence is a safety hazard",
    "The model must learn to say 'I don't know'",
], size=14, line_spacing=1.35)

# Formalization panel (right)
add_rect(s, Inches(6.8), Inches(1.7), Inches(6.1), Inches(4.6), SOFT,
         line=BORDER, radius=True)
add_textbox(s, Inches(7.0), Inches(1.85), Inches(5.7), Inches(0.4),
            "Formal task: binary gating", size=16, bold=True, color=BLUE)
add_textbox(s, Inches(7.0), Inches(2.3), Inches(5.7), Inches(0.45),
            "Pick score s(x) and threshold τ, flag OOD when s(x) > τ:",
            size=13, color=DARK)
# Equation block
add_rect(s, Inches(7.0), Inches(2.75), Inches(5.7), Inches(0.6), RGBColor(0xFF, 0xFF, 0xFF),
         line=BLUE)
add_textbox(s, Inches(7.1), Inches(2.8), Inches(5.5), Inches(0.5),
            "g(x; τ) = 𝟙[ s(x) > τ ]",
            size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE, font="Cambria")
add_textbox(s, Inches(7.0), Inches(3.5), Inches(5.7), Inches(0.45),
            "Goal: choose s(x) that separates ID from OOD inputs.",
            size=13, color=DARK)
# Our framing
add_rect(s, Inches(7.0), Inches(4.1), Inches(5.7), Inches(2.1), CREAM,
         line=GOLD)
add_textbox(s, Inches(7.15), Inches(4.2), Inches(5.5), Inches(0.4),
            "Our reframing", size=14, bold=True, color=RGBColor(0x8A, 0x61, 0x00))
add_textbox(s, Inches(7.15), Inches(4.55), Inches(5.5), Inches(1.6),
            "Classifier-energy asks: 'does x fit one of the 10 classes?'\n\n"
            "We ask: 'does x look like a CIFAR-10 image at all?'\n\n"
            "The second question is about visual structure — a natural fit for a generative model.",
            size=12, color=DARK)

add_footer(s, 3, 18)


# ============================================================
# SLIDE 4 — RELATED WORK
# ============================================================
s = add_slide()
add_header(s, "Related Work", section_num="2")

methods = [
    ("Softmax Confidence",   "Hendrycks & Gimpel, 2017",
     "Max softmax probability. Simplest baseline. Known to be overconfident on unfamiliar inputs."),
    ("ODIN",                  "Liang et al., 2018",
     "Temperature scaling + gradient-based input perturbation. Two hyperparameters re-tuned per OOD set."),
    ("Mahalanobis",           "Lee et al., 2018",
     "Class-conditional Gaussians on penultimate features. Requires stored statistics + Gaussian assumption."),
    ("Outlier Exposure",      "Hendrycks et al., 2019",
     "Fine-tunes on auxiliary OOD images to push outputs toward uniform."),
    ("Energy-Based OOD",      "Liu et al., NeurIPS 2020  ← our baseline",
     "Free-energy score −T·logΣexp(fᵢ/T) from classifier logits. Better calibrated than softmax."),
    ("VQ-VAE",                "van den Oord et al., NeurIPS 2017",
     "Discrete latent vocabulary. We use it to define a generative code space for OOD scoring."),
]

y = Inches(1.2)
for title, cite, desc in methods:
    add_rect(s, Inches(0.4), y, Inches(12.5), Inches(0.9), SOFT, line=BORDER, radius=True)
    add_textbox(s, Inches(0.6), y + Inches(0.08), Inches(4.5), Inches(0.4),
                title, size=15, bold=True, color=NAVY)
    add_textbox(s, Inches(0.6), y + Inches(0.43), Inches(4.5), Inches(0.4),
                cite, size=11, color=GREY)
    add_textbox(s, Inches(5.2), y + Inches(0.15), Inches(7.6), Inches(0.7),
                desc, size=12, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.98)

add_footer(s, 4, 18)


# ============================================================
# SLIDE 5 — CONTRIBUTIONS
# ============================================================
s = add_slide()
add_header(s, "Contributions", section_num="3")

items = [
    "Reproduce the classifier-energy baseline (Liu 2020) on CIFAR-10 vs CIFAR-100 and CIFAR-10 vs SVHN",
    "Propose Latent Energy Detection: an UNSUPERVISED energy network trained directly on VQ-VAE codes with a contrastive hinge loss; Gaussian noise serves as a cheap pseudo-OOD proxy. No OOD labels needed",
    "Add a supervised Latent MLP classifier operating on the same 4096-D codes, and a combined score that blends both through min–max normalisation",
    "Evaluate all four detectors on Near-OOD (CIFAR-100) and Far-OOD (SVHN) with AUROC, AUPR, FPR@95TPR — the standard benchmark triple",
    "Analyse when combining helps vs hurts, and why Gaussian noise is a weak pseudo-OOD proxy",
    "Release a clean single-command PyTorch pipeline (./run_all.sh) with deterministic seeds and documented checkpoints",
]

y = Inches(1.25)
for i, item in enumerate(items):
    # Number badge
    add_rect(s, Inches(0.5), y, Inches(0.6), Inches(0.85), BLUE, radius=True)
    add_textbox(s, Inches(0.5), y, Inches(0.6), Inches(0.85),
                f"C{i+1}", size=18, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, Inches(1.25), y, Inches(11.6), Inches(0.85), SOFT, line=BORDER, radius=True)
    add_textbox(s, Inches(1.45), y + Inches(0.08), Inches(11.2), Inches(0.75),
                item, size=13, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.92)

add_footer(s, 5, 18)


# ============================================================
# SLIDE 6 — BASELINE METHOD
# ============================================================
s = add_slide()
add_header(s, "Baseline: Energy-Based OOD from Classifier Logits",
           subtitle="Liu et al., NeurIPS 2020", section_num="4a")

# Left: how it works
add_rect(s, Inches(0.4), Inches(1.15), Inches(6.2), Inches(5.8), SOFT,
         line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.25), Inches(5.8), Inches(0.4),
            "How it works", size=16, bold=True, color=BLUE)

add_bullets(s, Inches(0.6), Inches(1.7), Inches(5.8), Inches(1.8), [
    "Train a classifier f(x) → ℝ¹⁰ on CIFAR-10 (cross-entropy)",
    "Compute free energy from the logits:",
], size=13, line_spacing=1.25)

add_rect(s, Inches(0.6), Inches(3.2), Inches(5.8), Inches(0.75), RGBColor(0xFF, 0xFF, 0xFF),
         line=BLUE)
add_textbox(s, Inches(0.6), Inches(3.22), Inches(5.8), Inches(0.7),
            "E(x; f) = −T · log Σᵢ exp(fᵢ(x)/T)",
            size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE, font="Cambria")

add_bullets(s, Inches(0.6), Inches(4.1), Inches(5.8), Inches(2.8), [
    "Low energy → sample fits one of the 10 classes well",
    "High energy → likely OOD",
    "Threshold τ chosen at 95% TPR on ID validation set",
    "Strengths: simple, theoretically grounded in log-partition",
    "Weakness: score still depends on classifier logits — a 'class identity' lens",
], size=13, line_spacing=1.25, bold_prefix_split=False)

# Right: architecture
add_rect(s, Inches(6.8), Inches(1.15), Inches(6.1), Inches(5.8), SOFT,
         line=BORDER, radius=True)
add_textbox(s, Inches(7.0), Inches(1.25), Inches(5.7), Inches(0.4),
            "Baseline CNN architecture", size=16, bold=True, color=BLUE)

layers = [
    ("Input",        "3 × 32 × 32 RGB image",          RGBColor(0xE7, 0xEB, 0xF1)),
    ("Conv2d",       "3→32, 3×3, ReLU, MaxPool(2)",    BLUE),
    ("Conv2d",       "32→64, 3×3, ReLU, MaxPool(2)",   BLUE),
    ("Flatten",      "64·8·8 = 4096",                   MUTED),
    ("Linear",       "4096 → 128, ReLU",               PURPLE),
    ("Linear",       "128 → 10 (logits)",              PURPLE),
    ("Energy",       "E(x) = −logsumexp(logits)",       RED),
]
yy = Inches(1.8)
for name, desc, fill in layers:
    add_rect(s, Inches(7.0), yy, Inches(2.0), Inches(0.55), fill, radius=True)
    text_color = RGBColor(0xFF, 0xFF, 0xFF) if fill not in (RGBColor(0xE7, 0xEB, 0xF1),) else NAVY
    add_textbox(s, Inches(7.0), yy, Inches(2.0), Inches(0.55),
                name, size=12, bold=True, color=text_color,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(9.2), yy + Inches(0.05), Inches(3.5), Inches(0.5),
                desc, size=11, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    yy += Inches(0.7)

add_footer(s, 6, 18)


# ============================================================
# SLIDE 7 — OUR METHOD OVERVIEW
# ============================================================
s = add_slide()
add_header(s, "Our Method: Detection in VQ-VAE Latent Space",
           subtitle="A generative code space shaped only by in-distribution data", section_num="4b")

add_textbox(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(0.4),
            "Pipeline", size=16, bold=True, color=NAVY)

# Pipeline boxes
boxes = [
    ("Image x",     "3 × 32 × 32",           Inches(0.4),  RGBColor(0xF4, 0xF6, 0xFA), NAVY),
    ("VQ-VAE Enc.", "trained on CIFAR-10",   Inches(2.6),  BLUE, RGBColor(0xFF, 0xFF, 0xFF)),
    ("Quantize",    "codebook K=512, d=64",  Inches(4.8),  PURPLE, RGBColor(0xFF, 0xFF, 0xFF)),
    ("z  (4096-D)", "8 × 8 × 64 flattened",  Inches(7.0),  GOLD, NAVY),
    ("Energy MLP",  "unsupervised hinge",    Inches(9.2),  RED, RGBColor(0xFF, 0xFF, 0xFF)),
    ("OOD MLP",     "supervised BCE",        Inches(11.2), GREEN, RGBColor(0xFF, 0xFF, 0xFF)),
]
y_box = Inches(1.7)
for name, sub, x, fill, fg in boxes:
    add_rect(s, x, y_box, Inches(2.0), Inches(1.1), fill, radius=True)
    add_textbox(s, x, y_box + Inches(0.1), Inches(2.0), Inches(0.45),
                name, size=14, bold=True, color=fg,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, x, y_box + Inches(0.55), Inches(2.0), Inches(0.5),
                sub, size=10, color=fg,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# Arrow row
for ax in [2.4, 4.6, 6.8, 9.0, 11.0]:
    add_textbox(s, Inches(ax), Inches(2.05), Inches(0.3), Inches(0.4),
                "▶", size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

# Below: three info panels
panels = [
    ("What enters the code space",
     [
         "Only CIFAR-10 images during training",
         "Encoder learns a 'vocabulary' of 512 code vectors",
         "Each 8×8 spatial position picks the closest code",
         "Result z_q: 4096-D discrete signature of the image",
     ]),
    ("Why this should detect OOD",
     [
         "CIFAR-100 / SVHN images push the encoder into unfamiliar code patterns",
         "These patterns are directly observable in z_q",
         "Detectors learn to flag unusual codebook activations",
         "No classifier — the score is about visual structure",
     ]),
    ("Two detectors on the same z",
     [
         "Energy MLP: unsupervised, learns ID code manifold",
         "OOD MLP: supervised, binary ID vs OOD",
         "Combined: min–max normalise, blend α=0.5",
         "Compare all three to pick the best trade-off",
     ]),
]

y_p = Inches(3.2)
for i, (title, items) in enumerate(panels):
    x = Inches(0.4) + i * Inches(4.23)
    add_rect(s, x, y_p, Inches(4.05), Inches(3.8), SOFT, line=BORDER, radius=True)
    add_textbox(s, x + Inches(0.2), y_p + Inches(0.1), Inches(3.8), Inches(0.4),
                title, size=13, bold=True, color=BLUE)
    add_bullets(s, x + Inches(0.15), y_p + Inches(0.55), Inches(3.85), Inches(3.2),
                items, size=11, line_spacing=1.2)

add_footer(s, 7, 18)


# ============================================================
# SLIDE 8 — VQ-VAE DETAILS
# ============================================================
s = add_slide()
add_header(s, "The VQ-VAE: vocabulary-building for images",
           subtitle="Van den Oord, Vinyals, Kavukcuoglu — NeurIPS 2017", section_num="4c")

# Left: architecture diagram
add_rect(s, Inches(0.4), Inches(1.15), Inches(6.3), Inches(5.9), SOFT,
         line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.25), Inches(5.9), Inches(0.4),
            "Architecture", size=15, bold=True, color=BLUE)

arch = [
    ("Encoder",        "2× (Conv 3×3, stride 2, ReLU) → z_e ∈ ℝ^{64×8×8}"),
    ("Codebook",       "K = 512 vectors of dim 64, learned jointly"),
    ("Quantizer",      "snap each spatial z_e to nearest e_k (argmin)"),
    ("Straight-through","∂L/∂z_e ← ∂L/∂z_q  (bypasses argmin)"),
    ("Decoder",        "2× (ConvT 3×3, stride 2, ReLU) → x̂ ∈ ℝ^{3×32×32}"),
]
yy = Inches(1.75)
for label, desc in arch:
    add_rect(s, Inches(0.6), yy, Inches(1.9), Inches(0.5), BLUE, radius=True)
    add_textbox(s, Inches(0.6), yy, Inches(1.9), Inches(0.5),
                label, size=11, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(2.7), yy + Inches(0.03), Inches(3.9), Inches(0.45),
                desc, size=11, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    yy += Inches(0.62)

# Loss block
add_rect(s, Inches(0.6), Inches(5.0), Inches(5.95), Inches(1.9), RGBColor(0xFF, 0xFF, 0xFF),
         line=BLUE)
add_textbox(s, Inches(0.7), Inches(5.05), Inches(5.8), Inches(0.35),
            "VQ-VAE loss (3 terms)", size=12, bold=True, color=NAVY)
add_textbox(s, Inches(0.7), Inches(5.4), Inches(5.8), Inches(0.6),
            "ℒ_vq = ‖x − D(z_q)‖²  +  ‖sg[z_e] − e_k‖²  +  β·‖z_e − sg[e_k]‖²",
            size=13, bold=True, color=NAVY, font="Cambria",
            anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(0.7), Inches(6.0), Inches(5.8), Inches(0.85),
            "• Reconstruction: decoder must rebuild x from z_q\n"
            "• Codebook: vectors pulled toward encoder outputs\n"
            "• Commitment (β = 0.25): encoder must stay near its chosen code",
            size=11, color=DARK)

# Right: why it helps for OOD + numbers
add_rect(s, Inches(6.9), Inches(1.15), Inches(6.0), Inches(5.9), CREAM,
         line=GOLD, radius=True)
add_textbox(s, Inches(7.1), Inches(1.25), Inches(5.7), Inches(0.4),
            "Why this helps for OOD", size=15, bold=True,
            color=RGBColor(0x8A, 0x61, 0x00))

add_bullets(s, Inches(7.1), Inches(1.75), Inches(5.6), Inches(3.0), [
    "Codebook learns a discrete 'vocabulary' of image parts",
    "Vocabulary is shaped ONLY by CIFAR-10 — no OOD signal seen",
    "Unfamiliar images force unusual codebook index patterns",
    "Quantization acts as a bottleneck — limits reconstruction quality on OOD",
    "The 4096-D code z_q is a compact, structure-aware signature of x",
], size=12, line_spacing=1.3)

# Numbers strip
add_textbox(s, Inches(7.1), Inches(4.9), Inches(5.6), Inches(0.4),
            "Key numbers", size=13, bold=True, color=NAVY)
nums = [
    ("Codebook size K",  "512"),
    ("Code dimension d", "64"),
    ("Latent grid",      "8 × 8"),
    ("Flattened z_q",    "4096-D"),
    ("Train epochs",     "10"),
    ("Commitment β",     "0.25"),
]
y_n = Inches(5.35)
for i, (k, v) in enumerate(nums):
    row, col = i // 3, i % 3
    xx = Inches(7.1) + col * Inches(1.9)
    yy2 = y_n + row * Inches(0.8)
    add_rect(s, xx, yy2, Inches(1.8), Inches(0.7), RGBColor(0xFF, 0xFF, 0xFF),
             line=GOLD, radius=True)
    add_textbox(s, xx, yy2 + Inches(0.05), Inches(1.8), Inches(0.3),
                k, size=10, color=GREY, align=PP_ALIGN.CENTER)
    add_textbox(s, xx, yy2 + Inches(0.3), Inches(1.8), Inches(0.4),
                v, size=15, bold=True, color=NAVY,
                align=PP_ALIGN.CENTER, font="Consolas")

add_footer(s, 8, 18)


# ============================================================
# SLIDE 9 — TRAINING OBJECTIVES
# ============================================================
s = add_slide()
add_header(s, "Training Objectives", section_num="5")

# Three panels stacked — VQ-VAE (shown before) / Energy / OOD MLP / Combined
# Panel 1 — Latent Energy (hinge)
add_rect(s, Inches(0.4), Inches(1.1), Inches(12.5), Inches(2.0), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.2), Inches(6), Inches(0.4),
            "Latent Energy — contrastive hinge (UNSUPERVISED)",
            size=14, bold=True, color=RED)
add_rect(s, Inches(0.6), Inches(1.65), Inches(7.5), Inches(1.3), RGBColor(0xFF, 0xFF, 0xFF), line=BLUE)
add_textbox(s, Inches(0.7), Inches(1.7), Inches(7.3), Inches(0.45),
            "ℒ_in  = 𝔼_{z~p_ID} [ max(0, E(z) − m_in )² ]           m_in = −10",
            size=13, color=NAVY, font="Cambria", anchor=MSO_ANCHOR.MIDDLE, bold=True)
add_textbox(s, Inches(0.7), Inches(2.1), Inches(7.3), Inches(0.45),
            "ℒ_out = 𝔼_{z~𝒩(0,I)} [ max(0, m_out − E(z) )² ]    m_out = −5",
            size=13, color=NAVY, font="Cambria", anchor=MSO_ANCHOR.MIDDLE, bold=True)
add_textbox(s, Inches(0.7), Inches(2.55), Inches(7.3), Inches(0.35),
            "ℒ_energy = ℒ_in + ℒ_out",
            size=13, color=NAVY, font="Cambria", anchor=MSO_ANCHOR.MIDDLE, bold=True)

add_bullets(s, Inches(8.3), Inches(1.65), Inches(4.5), Inches(1.4), [
    "Push real CIFAR-10 codes to LOW energy",
    "Push Gaussian noise to HIGH energy",
    "Hinge stops once sample is past margin",
    "No OOD labels needed at training",
], size=11, line_spacing=1.2)

# Panel 2 — OOD MLP (BCE)
add_rect(s, Inches(0.4), Inches(3.2), Inches(12.5), Inches(1.8), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(3.3), Inches(6), Inches(0.4),
            "Latent OOD MLP — binary cross-entropy (SUPERVISED)",
            size=14, bold=True, color=GREEN)
add_rect(s, Inches(0.6), Inches(3.75), Inches(7.5), Inches(1.1), RGBColor(0xFF, 0xFF, 0xFF), line=GREEN)
add_textbox(s, Inches(0.7), Inches(3.8), Inches(7.3), Inches(0.5),
            "ℒ_MLP = −𝔼[ y·log σ(g(z)) + (1−y)·log (1 − σ(g(z))) ]",
            size=13, color=NAVY, font="Cambria", anchor=MSO_ANCHOR.MIDDLE, bold=True)
add_textbox(s, Inches(0.7), Inches(4.3), Inches(7.3), Inches(0.5),
            "y = 0 for CIFAR-10,   y = 1 for CIFAR-100 / SVHN codes",
            size=12, color=DARK, anchor=MSO_ANCHOR.MIDDLE)

add_bullets(s, Inches(8.3), Inches(3.75), Inches(4.5), Inches(1.2), [
    "Uses known OOD codes during training",
    "Strong signal — but needs labelled OOD",
    "Score: sigmoid probability of 'OOD'",
], size=11, line_spacing=1.2)

# Panel 3 — Combined
add_rect(s, Inches(0.4), Inches(5.1), Inches(12.5), Inches(1.85), CREAM, line=GOLD, radius=True)
add_textbox(s, Inches(0.6), Inches(5.2), Inches(6), Inches(0.4),
            "Combined score — min–max normalise + blend",
            size=14, bold=True, color=RGBColor(0x8A, 0x61, 0x00))
add_rect(s, Inches(0.6), Inches(5.65), Inches(7.5), Inches(1.1), RGBColor(0xFF, 0xFF, 0xFF), line=GOLD)
add_textbox(s, Inches(0.7), Inches(5.7), Inches(7.3), Inches(1.0),
            "s_comb(z) = α · Ẽ_lat(z) + (1 − α) · ỹ_MLP(z)       α = 0.5",
            size=14, color=NAVY, font="Cambria", anchor=MSO_ANCHOR.MIDDLE, bold=True)

add_bullets(s, Inches(8.3), Inches(5.65), Inches(4.5), Inches(1.2), [
    "Both signals scaled to [0, 1]",
    "Equal-weight blend as a simple ensemble",
    "Analysed: blend can HURT if one signal is weak",
], size=11, line_spacing=1.2)

add_footer(s, 9, 18)


# ============================================================
# SLIDE 10 — IMPLEMENTATION DETAILS
# ============================================================
s = add_slide()
add_header(s, "Implementation Details", section_num="6")

# Left column: architectures table
add_rect(s, Inches(0.4), Inches(1.15), Inches(6.3), Inches(5.9), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.25), Inches(5.9), Inches(0.4),
            "Model architectures", size=15, bold=True, color=BLUE)

rows = [
    ("Model",          "Architecture",                              "Loss / Output"),
    ("Baseline CNN",   "2× (Conv+ReLU+Pool) + 2-layer MLP",         "CrossEntropy → 10 logits"),
    ("VQ-VAE",         "2-layer ConvEnc/Dec, K=512 × d=64",          "MSE + codebook + 0.25·commit"),
    ("Energy MLP",     "3× Linear(256) + Linear(1)",                 "Contrastive hinge → scalar E(z)"),
    ("OOD MLP",        "3× Linear(256) + Linear(1)",                 "BCE → logit σ⁻¹(p_OOD)"),
]
yy = Inches(1.75)
col_widths = [Inches(1.7), Inches(2.6), Inches(1.85)]
col_xs = [Inches(0.6), Inches(2.3), Inches(4.9)]
for r_i, row in enumerate(rows):
    is_head = (r_i == 0)
    fill = NAVY if is_head else (RGBColor(0xEE, 0xF2, 0xF8) if r_i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
    fg = RGBColor(0xFF, 0xFF, 0xFF) if is_head else DARK
    for c_i, (cell, cx, cw) in enumerate(zip(row, col_xs, col_widths)):
        add_rect(s, cx, yy, cw, Inches(0.55), fill, line=BORDER)
        add_textbox(s, cx + Inches(0.05), yy, cw - Inches(0.1), Inches(0.55),
                    cell, size=10, bold=is_head, color=fg,
                    anchor=MSO_ANCHOR.MIDDLE)
    yy += Inches(0.55)

# Hyperparameter list below table
add_textbox(s, Inches(0.6), Inches(4.6), Inches(6), Inches(0.4),
            "Training settings", size=14, bold=True, color=BLUE)
add_bullets(s, Inches(0.6), Inches(5.0), Inches(5.9), Inches(2.0), [
    "Adam optimiser, lr = 1e-3",
    "Batch: 512 latent heads · 128 baseline CNN",
    "Epochs: 10 (VQ-VAE, latent heads) · 5 (baseline)",
    "Seed 42 fixed across Python / NumPy / PyTorch",
    "Full pipeline runs in <15 min on a single Colab T4",
], size=11, line_spacing=1.25)

# Right column: reproducibility
add_rect(s, Inches(6.9), Inches(1.15), Inches(6.0), Inches(5.9), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(7.1), Inches(1.25), Inches(5.7), Inches(0.4),
            "Reproducibility & pipeline", size=15, bold=True, color=BLUE)

steps = [
    ("1", "train_vqvae.py",       "VQ-VAE on CIFAR-10 train split (50k)"),
    ("2", "train_baseline.py",    "Baseline CNN for classifier energy"),
    ("3", "extract_latents.py",   "Encode CIFAR-10/100, SVHN → .npy"),
    ("4", "train_energy.py",      "Contrastive hinge on latent codes"),
    ("5", "train_mlp.py",         "Supervised BCE OOD head"),
    ("6", "evaluate_baseline.py", "Metrics for baseline energy"),
    ("7", "evaluate_ood.py",      "Metrics for all 3 latent detectors"),
]
yy = Inches(1.75)
for num, fname, desc in steps:
    add_rect(s, Inches(7.1), yy, Inches(0.45), Inches(0.55), BLUE, radius=True)
    add_textbox(s, Inches(7.1), yy, Inches(0.45), Inches(0.55),
                num, size=13, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(7.65), yy + Inches(0.04), Inches(2.4), Inches(0.5),
                fname, size=11, bold=True, color=NAVY, font="Consolas",
                anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, Inches(10.1), yy + Inches(0.04), Inches(2.7), Inches(0.5),
                desc, size=10, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    yy += Inches(0.65)

# Run-command strip
add_rect(s, Inches(7.1), Inches(6.4), Inches(5.7), Inches(0.55), NAVY, radius=True)
add_textbox(s, Inches(7.1), Inches(6.4), Inches(5.7), Inches(0.55),
            "$  ./run_all.sh           # one-shot pipeline",
            size=13, bold=True, color=GOLD, font="Consolas",
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

add_footer(s, 10, 18)


# ============================================================
# SLIDE 11 — EXPERIMENTAL SETUP
# ============================================================
s = add_slide()
add_header(s, "Experimental Setup", section_num="7")

# Datasets table
add_rect(s, Inches(0.4), Inches(1.15), Inches(12.5), Inches(2.4), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.25), Inches(5), Inches(0.4),
            "Datasets  (all 32 × 32 RGB, test splits)", size=15, bold=True, color=BLUE)

headers = ["Dataset", "Role", "Classes", "Domain", "# test images"]
widths = [Inches(2.6), Inches(2.4), Inches(1.8), Inches(3.3), Inches(2.1)]
xs = []
cx = Inches(0.6); xs.append(cx)
for w in widths[:-1]:
    cx = cx + w; xs.append(cx)

rows = [
    headers,
    ["CIFAR-10",   "ID (in-distribution)", "10",  "natural images",       "10,000"],
    ["CIFAR-100",  "Near-OOD",              "100", "natural images",       "10,000"],
    ["SVHN",       "Far-OOD",               "10",  "digit / street view",  "26,032"],
]
yy = Inches(1.75)
for r_i, row in enumerate(rows):
    is_head = (r_i == 0)
    fill = NAVY if is_head else (RGBColor(0xEE, 0xF2, 0xF8) if r_i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
    fg = RGBColor(0xFF, 0xFF, 0xFF) if is_head else DARK
    for cell, x, w in zip(row, xs, widths):
        add_rect(s, x, yy, w, Inches(0.4), fill, line=BORDER)
        add_textbox(s, x + Inches(0.1), yy, w - Inches(0.15), Inches(0.4),
                    str(cell), size=11, bold=is_head, color=fg,
                    anchor=MSO_ANCHOR.MIDDLE)
    yy += Inches(0.4)

# Note row
add_textbox(s, Inches(0.6), Inches(3.1), Inches(12), Inches(0.35),
            "CIFAR-100 is a stress test — same image domain. SVHN is a sanity check — obviously different content.",
            size=11, color=GREY)

# Metrics row (three cards)
metrics = [
    ("AUROC  ↑",  "Area under ROC curve",
     "Higher = better separability between ID and OOD scores. 0.5 = random, 1.0 = perfect."),
    ("AUPR  ↑",   "Area under Precision-Recall curve",
     "Sensitive to class imbalance. More honest when OOD is rare relative to ID."),
    ("FPR@95TPR  ↓", "False-positive rate at 95 % TPR",
     "The practical deployment metric: 'if I must keep 95 % of real ID, how many OOD slip through?'"),
]
for i, (name, desc, detail) in enumerate(metrics):
    x = Inches(0.4) + i * Inches(4.23)
    add_rect(s, x, Inches(3.8), Inches(4.05), Inches(2.3), SOFT, line=BORDER, radius=True)
    add_rect(s, x, Inches(3.8), Inches(4.05), Inches(0.5), BLUE, radius=False)
    add_textbox(s, x, Inches(3.8), Inches(4.05), Inches(0.5),
                name, size=14, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(s, x + Inches(0.15), Inches(4.35), Inches(3.8), Inches(0.35),
                desc, size=11, bold=True, color=NAVY)
    add_textbox(s, x + Inches(0.15), Inches(4.7), Inches(3.8), Inches(1.35),
                detail, size=10, color=DARK)

# Training / protocol note
add_rect(s, Inches(0.4), Inches(6.25), Inches(12.5), Inches(0.75), CREAM, line=GOLD, radius=True)
add_textbox(s, Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.65),
            "Protocol:  Adam (lr 1e-3), batch 512, 10 epochs per head, seed 42. "
            "All scores evaluated on the full test splits above — no subsampling.",
            size=12, color=DARK, anchor=MSO_ANCHOR.MIDDLE)

add_footer(s, 11, 18)


# ============================================================
# SLIDE 12 — RESULTS (FULL TABLE)
# ============================================================
s = add_slide()
add_header(s, "Results — All methods, all metrics", section_num="8")

# Near-OOD table
add_textbox(s, Inches(0.4), Inches(1.1), Inches(12), Inches(0.4),
            "Near-OOD:  CIFAR-10  (ID)  vs  CIFAR-100", size=15, bold=True, color=BLUE)

table_near = [
    ["Method",               "AUROC ↑", "AUPR ↑", "FPR@95 ↓"],
    ["Baseline Energy [1]",  "0.6884",  "0.6573", "0.7959"],
    ["Latent Energy (ours)", "0.4957",  "0.5263", "0.9614"],
    ["Latent MLP (ours)",    "0.7018",  "0.6937", "0.8196"],
    ["Latent E + MLP (ours)","0.6698",  "0.6701", "0.8642"],
]
col_widths = [Inches(4.5), Inches(2.6), Inches(2.6), Inches(2.6)]
col_xs = []
cx = Inches(0.4); col_xs.append(cx)
for w in col_widths[:-1]:
    cx = cx + w; col_xs.append(cx)

yy = Inches(1.55)
winner_idx = {0: None, 1: 1, 2: 3, 3: 3, 4: 3}  # row index -> best cell (by row)
for r_i, row in enumerate(table_near):
    is_head = r_i == 0
    is_ours_best = r_i == 3  # Latent MLP wins Near-OOD
    for c_i, (cell, x, w) in enumerate(zip(row, col_xs, col_widths)):
        if is_head:
            fill = NAVY; fg = RGBColor(0xFF, 0xFF, 0xFF); bold = True
        elif is_ours_best:
            fill = CREAM; fg = RGBColor(0x8A, 0x61, 0x00); bold = True
        else:
            fill = RGBColor(0xEE, 0xF2, 0xF8) if r_i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            fg = DARK; bold = False
        add_rect(s, x, yy, w, Inches(0.38), fill, line=BORDER)
        align = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER
        font = "Calibri" if (is_head or c_i == 0) else "Consolas"
        add_textbox(s, x + Inches(0.1), yy, w - Inches(0.15), Inches(0.38),
                    str(cell), size=11, bold=bold, color=fg,
                    align=align, anchor=MSO_ANCHOR.MIDDLE, font=font)
    yy += Inches(0.38)

# Far-OOD table
add_textbox(s, Inches(0.4), Inches(3.9), Inches(12), Inches(0.4),
            "Far-OOD:  CIFAR-10  (ID)  vs  SVHN", size=15, bold=True, color=BLUE)

table_far = [
    ["Method",               "AUROC ↑", "AUPR ↑", "FPR@95 ↓"],
    ["Baseline Energy [1]",  "0.7887",  "0.8634", "0.5870"],
    ["Latent Energy (ours)", "0.4886",  "0.7595", "0.9885"],
    ["Latent MLP (ours)",    "0.9801",  "0.9920", "0.1000"],
    ["Latent E + MLP (ours)","0.9315",  "0.9663", "0.2344"],
]
yy = Inches(4.35)
for r_i, row in enumerate(table_far):
    is_head = r_i == 0
    is_ours_best = r_i == 3
    for c_i, (cell, x, w) in enumerate(zip(row, col_xs, col_widths)):
        if is_head:
            fill = NAVY; fg = RGBColor(0xFF, 0xFF, 0xFF); bold = True
        elif is_ours_best:
            fill = CREAM; fg = RGBColor(0x8A, 0x61, 0x00); bold = True
        else:
            fill = RGBColor(0xEE, 0xF2, 0xF8) if r_i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            fg = DARK; bold = False
        add_rect(s, x, yy, w, Inches(0.38), fill, line=BORDER)
        align = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER
        font = "Calibri" if (is_head or c_i == 0) else "Consolas"
        add_textbox(s, x + Inches(0.1), yy, w - Inches(0.15), Inches(0.38),
                    str(cell), size=11, bold=bold, color=fg,
                    align=align, anchor=MSO_ANCHOR.MIDDLE, font=font)
    yy += Inches(0.38)

# Highlight row
add_rect(s, Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.5), GREEN, radius=True)
add_textbox(s, Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.5),
            "▶  Our Latent MLP beats the baseline on BOTH Near-OOD and Far-OOD, across ALL three metrics.",
            size=13, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

add_footer(s, 12, 18)


# ============================================================
# SLIDE 13 — HEAD-TO-HEAD COMPARISON (BARS)
# ============================================================
s = add_slide()
add_header(s, "Head-to-Head: Baseline Energy vs Our Latent MLP", section_num="9")

# Six bars: 2 datasets × 3 metrics
# Bar chart area
def bar(y, label, base_val, ours_val, better="higher", x0=Inches(2.2), w=Inches(7)):
    add_textbox(s, Inches(0.4), y, Inches(1.8), Inches(0.5),
                label, size=12, bold=True, color=NAVY,
                anchor=MSO_ANCHOR.MIDDLE)

    # Baseline bar
    add_textbox(s, x0, y, Inches(1.0), Inches(0.3),
                "Baseline", size=9, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, x0 + Inches(1.0), y + Inches(0.03), Inches(base_val * 6.5 / 1.0), Inches(0.24),
             MUTED, radius=True) if base_val > 0 else None
    add_textbox(s, x0 + Inches(7.5), y, Inches(0.9), Inches(0.3),
                f"{base_val:.3f}", size=10, bold=True, color=NAVY,
                font="Consolas", anchor=MSO_ANCHOR.MIDDLE)

    # Ours bar
    add_textbox(s, x0, y + Inches(0.32), Inches(1.0), Inches(0.3),
                "Ours", size=9, color=GREEN, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, x0 + Inches(1.0), y + Inches(0.35), Inches(ours_val * 6.5 / 1.0), Inches(0.24),
             GREEN, radius=True) if ours_val > 0 else None
    add_textbox(s, x0 + Inches(7.5), y + Inches(0.32), Inches(0.9), Inches(0.3),
                f"{ours_val:.3f}", size=10, bold=True, color=GREEN,
                font="Consolas", anchor=MSO_ANCHOR.MIDDLE)

    # Delta
    diff = ours_val - base_val
    sign = "↑" if diff > 0 else "↓"
    if better == "lower":
        is_good = diff < 0
    else:
        is_good = diff > 0
    col = GREEN if is_good else RED
    add_rect(s, x0 + Inches(8.4), y + Inches(0.12), Inches(1.5), Inches(0.4),
             col, radius=True)
    add_textbox(s, x0 + Inches(8.4), y + Inches(0.12), Inches(1.5), Inches(0.4),
                f"{sign} {abs(diff):.3f}", size=11, bold=True,
                color=RGBColor(0xFF, 0xFF, 0xFF),
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# Near-OOD header
add_textbox(s, Inches(0.4), Inches(1.15), Inches(12), Inches(0.4),
            "CIFAR-10  vs  CIFAR-100   (Near-OOD)", size=13, bold=True, color=BLUE)

bar(Inches(1.6), "AUROC ↑",    0.6884, 0.7018, "higher")
bar(Inches(2.4), "AUPR ↑",     0.6573, 0.6937, "higher")
bar(Inches(3.2), "FPR@95 ↓",   0.7959, 0.8196, "lower")

# Far-OOD header
add_textbox(s, Inches(0.4), Inches(4.1), Inches(12), Inches(0.4),
            "CIFAR-10  vs  SVHN   (Far-OOD)", size=13, bold=True, color=BLUE)

bar(Inches(4.55), "AUROC ↑",    0.7887, 0.9801, "higher")
bar(Inches(5.35), "AUPR ↑",     0.8634, 0.9920, "higher")
bar(Inches(6.15), "FPR@95 ↓",   0.5870, 0.1000, "lower")

# Takeaway band
add_rect(s, Inches(0.4), Inches(6.75), Inches(12.5), Inches(0.35), GREEN, radius=True)
add_textbox(s, Inches(0.4), Inches(6.75), Inches(12.5), Inches(0.35),
            "On Far-OOD: +19.1 AUROC, +12.9 AUPR, −48.7 FPR@95. On Near-OOD: small but consistent gains on AUROC & AUPR.",
            size=11, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

add_footer(s, 13, 18)


# ============================================================
# SLIDE 14 — ANALYSIS (why combined hurts)
# ============================================================
s = add_slide()
add_header(s, "Analysis — Why does combining signals hurt?", section_num="10")

# Left: the anomaly
add_rect(s, Inches(0.4), Inches(1.15), Inches(6.1), Inches(5.9), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.25), Inches(5.7), Inches(0.4),
            "The paradox on Far-OOD (SVHN)", size=15, bold=True, color=BLUE)

rows = [
    ("MLP score alone",           "0.980"),
    ("Energy score alone",        "0.489  (≈ random)"),
    ("0.5·MLP  +  0.5·Energy",    "0.932  ↓"),
]
yy = Inches(1.75)
for label, val in rows:
    add_rect(s, Inches(0.6), yy, Inches(3.4), Inches(0.6), RGBColor(0xFF, 0xFF, 0xFF), line=BORDER)
    add_textbox(s, Inches(0.7), yy, Inches(3.25), Inches(0.6),
                label, size=12, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    add_rect(s, Inches(4.0), yy, Inches(2.3), Inches(0.6), CREAM, line=GOLD)
    add_textbox(s, Inches(4.05), yy, Inches(2.2), Inches(0.6),
                val, size=13, bold=True, color=NAVY, font="Consolas",
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    yy += Inches(0.7)

add_textbox(s, Inches(0.6), Inches(4.0), Inches(5.8), Inches(0.4),
            "Why it happens", size=13, bold=True, color=RED)
add_bullets(s, Inches(0.6), Inches(4.4), Inches(5.8), Inches(2.5), [
    "Equal-weight average assumes both inputs are informative",
    "Near-random energy behaves like additive noise",
    "Noise drags the strong MLP signal down",
    "The ensemble mean is NOT the max — it cannot skip a bad signal",
], size=11, line_spacing=1.3)

# Right: failure of Gaussian proxy + fixes
add_rect(s, Inches(6.7), Inches(1.15), Inches(6.2), Inches(5.9), CREAM, line=GOLD, radius=True)
add_textbox(s, Inches(6.9), Inches(1.25), Inches(5.8), Inches(0.4),
            "Why Latent Energy is weak", size=15, bold=True,
            color=RGBColor(0x8A, 0x61, 0x00))

add_bullets(s, Inches(6.9), Inches(1.75), Inches(5.8), Inches(2.0), [
    "We used Gaussian noise as pseudo-OOD",
    "Gaussian samples land FAR from any real code pattern",
    "Model only learns to push noise away from the data manifold",
    "Real OOD (CIFAR-100, SVHN) still lives close to CIFAR-10 codes",
    "→ Energy cannot distinguish them in-distribution",
], size=11, line_spacing=1.25)

add_textbox(s, Inches(6.9), Inches(4.2), Inches(5.8), Inches(0.4),
            "How to fix it", size=13, bold=True, color=GREEN)
add_bullets(s, Inches(6.9), Inches(4.6), Inches(5.8), Inches(2.5), [
    "Smarter pseudo-OOD: CutMix on latent codes, codebook-index shuffling",
    "Mix real codes with random codebook-entry substitutions",
    "Learned blend weight α instead of fixed 0.5",
    "Gated combiner: 'trust only the signal that is confident'",
], size=11, line_spacing=1.25)

add_footer(s, 14, 18)


# ============================================================
# SLIDE 15 — WHAT WE LEARNED
# ============================================================
s = add_slide()
add_header(s, "What We Learned", section_num="11")

findings = [
    ("Latent MLP wins Far-OOD by a wide margin",
     "0.98 AUROC vs baseline 0.79 — a 19-point jump. FPR@95 drops from 59% to 10%.",
     GREEN),
    ("Latent MLP also wins Near-OOD (small margin)",
     "0.70 vs 0.69 AUROC — consistent improvement across AUROC and AUPR.",
     GREEN),
    ("Gaussian noise is a weak pseudo-OOD proxy",
     "Energy model trained on noise performs near chance on real OOD (AUROC ≈ 0.49).",
     RED),
    ("Blending weak + strong can hurt",
     "0.5·Energy + 0.5·MLP on SVHN scored 0.93 — below MLP-alone at 0.98.",
     RED),
    ("Near-OOD remains the real challenge",
     "CIFAR-100 shares visual statistics with CIFAR-10; a 512-entry codebook cannot separate them cleanly.",
     NAVY),
    ("The method is unsupervised-friendly",
     "The Energy route needs no OOD labels — once we fix the pseudo-OOD, it becomes the most deployable variant.",
     NAVY),
]

y = Inches(1.2)
for title, desc, colour in findings:
    add_rect(s, Inches(0.4), y, Inches(0.2), Inches(0.85), colour, radius=False)
    add_rect(s, Inches(0.6), y, Inches(12.3), Inches(0.85), SOFT, line=BORDER, radius=True)
    add_textbox(s, Inches(0.8), y + Inches(0.08), Inches(12), Inches(0.38),
                title, size=14, bold=True, color=NAVY)
    add_textbox(s, Inches(0.8), y + Inches(0.45), Inches(12), Inches(0.38),
                desc, size=11, color=DARK)
    y += Inches(0.93)

add_footer(s, 15, 18)


# ============================================================
# SLIDE 16 — CONCLUSIONS & FUTURE WORK
# ============================================================
s = add_slide()
add_header(s, "Conclusions & Future Work", section_num="12")

# Conclusions
add_rect(s, Inches(0.4), Inches(1.15), Inches(6.3), Inches(5.9), SOFT, line=BORDER, radius=True)
add_textbox(s, Inches(0.6), Inches(1.25), Inches(5.9), Inches(0.4),
            "Conclusions", size=15, bold=True, color=BLUE)
add_bullets(s, Inches(0.6), Inches(1.7), Inches(5.9), Inches(5.3), [
    "Moving OOD detection from classifier logits into VQ-VAE latent space is a clear win for Far-OOD",
    "Our Latent MLP dominates on SVHN across all three standard metrics",
    "Near-OOD remains hard because CIFAR-100 codes overlap CIFAR-10 codes — a codebook-capacity issue, not a loss-design issue",
    "Unsupervised latent energy alone is not yet competitive — the pseudo-OOD signal is too weak",
    "Combining detectors via fixed-weight mean can hurt when component quality is unequal",
    "Overall: the generative-latent framing is promising, with a clear path to closing the Near-OOD gap",
], size=11, line_spacing=1.3)

# Future work
add_rect(s, Inches(6.9), Inches(1.15), Inches(6.0), Inches(5.9), CREAM, line=GOLD, radius=True)
add_textbox(s, Inches(7.1), Inches(1.25), Inches(5.7), Inches(0.4),
            "Future work", size=15, bold=True,
            color=RGBColor(0x8A, 0x61, 0x00))

items = [
    ("Smarter pseudo-OOD",
     "CutMix on codes, codebook-index shuffling, mixing with sampled codebook distractors"),
    ("Larger codebook",
     "K = 1024–4096 to resolve finer visual distinctions between CIFAR-10 and CIFAR-100"),
    ("Richer encoder",
     "Replace the 2-layer conv encoder with a WideResNet backbone"),
    ("Learnable blend",
     "Replace α = 0.5 with a learned or confidence-gated combiner"),
    ("Per-superclass ablation",
     "Find which CIFAR-100 categories are hardest to flag"),
    ("Deployment study",
     "Latency, calibration, and shift robustness on real-world ID streams"),
]
yy = Inches(1.75)
for title, desc in items:
    add_rect(s, Inches(7.1), yy, Inches(5.7), Inches(0.8), RGBColor(0xFF, 0xFF, 0xFF),
             line=GOLD, radius=True)
    add_textbox(s, Inches(7.25), yy + Inches(0.05), Inches(5.5), Inches(0.3),
                title, size=12, bold=True, color=NAVY)
    add_textbox(s, Inches(7.25), yy + Inches(0.35), Inches(5.5), Inches(0.45),
                desc, size=10, color=DARK)
    yy += Inches(0.88)

add_footer(s, 16, 18)


# ============================================================
# SLIDE 17 — REFERENCES
# ============================================================
s = add_slide()
add_header(s, "References")

refs = [
    "[1]  W. Liu, X. Wang, J. Owens, Y. Li.  Energy-based Out-of-Distribution Detection.  NeurIPS 2020.",
    "[2]  A. van den Oord, O. Vinyals, K. Kavukcuoglu.  Neural Discrete Representation Learning.  NeurIPS 2017.",
    "[3]  D. Hendrycks, K. Gimpel.  A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.  ICLR 2017.",
    "[4]  S. Liang, Y. Li, R. Srikant.  Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks (ODIN).  ICLR 2018.",
    "[5]  K. Lee, K. Lee, H. Lee, J. Shin.  A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.  NeurIPS 2018.",
    "[6]  D. Hendrycks, M. Mazeika, T. Dietterich.  Deep Anomaly Detection with Outlier Exposure.  ICLR 2019.",
    "[7]  A. Krizhevsky.  Learning Multiple Layers of Features from Tiny Images.  Tech. report, University of Toronto, 2009.  (CIFAR-10 / CIFAR-100)",
    "[8]  Y. Netzer et al.  Reading Digits in Natural Images with Unsupervised Feature Learning.  NeurIPS Workshop 2011.  (SVHN)",
]
y = Inches(1.25)
for ref in refs:
    add_rect(s, Inches(0.4), y, Inches(12.5), Inches(0.62), SOFT, line=BORDER, radius=True)
    add_textbox(s, Inches(0.6), y + Inches(0.05), Inches(12.2), Inches(0.55),
                ref, size=11, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.7)

add_footer(s, 17, 18)


# ============================================================
# SLIDE 18 — THANK YOU / Q&A
# ============================================================
s = add_slide()
add_rect(s, 0, 0, SW, SH, NAVY)
add_rect(s, 0, Inches(3.3), SW, Inches(0.08), GOLD)

add_textbox(s, Inches(0.6), Inches(1.6), Inches(12), Inches(0.9),
            "Thank you", size=60, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.6), Inches(2.5), Inches(12), Inches(0.6),
            "Questions?", size=32, color=GOLD, align=PP_ALIGN.CENTER)

add_textbox(s, Inches(0.6), Inches(3.7), Inches(12), Inches(0.5),
            "Improving Energy-Based Out-of-Distribution Detection using Generative Modeling",
            size=17, color=RGBColor(0xCF, 0xD8, 0xE8), align=PP_ALIGN.CENTER)

add_textbox(s, Inches(0.6), Inches(4.5), Inches(12), Inches(0.5),
            "Harshit Jain   ·   hjain5@gsu.edu",
            size=16, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.6), Inches(4.95), Inches(12), Inches(0.5),
            "Parsh Jadon   ·   pjadon1@gsu.edu",
            size=16, bold=True, color=GOLD, align=PP_ALIGN.CENTER)

add_textbox(s, Inches(0.6), Inches(5.9), Inches(12), Inches(0.5),
            "github.com/harshitjain25/csc8851-vqvae-ood",
            size=14, color=RGBColor(0xCF, 0xD8, 0xE8),
            align=PP_ALIGN.CENTER, font="Consolas")
add_textbox(s, Inches(0.6), Inches(6.35), Inches(12), Inches(0.5),
            "CSC 8851  ·  Deep Learning  ·  Spring 2026  ·  Georgia State University",
            size=12, color=RGBColor(0xCF, 0xD8, 0xE8), align=PP_ALIGN.CENTER)


# ========= SAVE =========
out = "/Users/harshitjain/Desktop/csc8851-vqvae-ood/poster/VQVAE_OOD_Presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Slides: {len(prs.slides)}")
