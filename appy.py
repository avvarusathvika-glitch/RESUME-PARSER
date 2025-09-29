# 📑 AI Resume Parser — Pastel Light Theme + EasyOCR + BERT NER + SBERT Skills + JD Fit + Timeline + Suggestions
# Fully self-contained Streamlit app.

import streamlit as st
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from PIL import Image
import pytesseract
import shutil
import re
import nltk
import json
import html
import math
import pandas as pd
import numpy as np
import altair as alt
import easyocr
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime
import dateparser
from dateutil.relativedelta import relativedelta

# ---------------- NLTK ----------------
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# ---------------- Page + Theme ----------------
st.set_page_config(layout="wide", page_title="AI Resume Parser", page_icon="📑")
st.markdown("""
<style>
:root{
  --bg:#ffffff;
  --ink:#0f172a;
  --muted:#f6f9ff;
  --brand:#3b82f6;
  --brand-2:#93c5fd;
  --brand-3:#dbeafe;
  --accent:#a5b4fc;
  --ink-2:#334155;
  --shadow:0 10px 30px rgba(30,64,175,.06);
}
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 10% -10%, var(--brand-3) 0%, transparent 40%),
              radial-gradient(900px 500px at 110% 10%, #e0f2fe 0%, transparent 35%),
              var(--bg) !important;
  color: var(--ink) !important;
}
h1,h2,h3,h4,h5,h6,label,p,span,div,small,strong,em,b,i { color: var(--ink) !important; }

/* bigger app title */
.app-title{ font-size:2.6rem; line-height:1.1; margin:0 0 .35rem 0; letter-spacing:.2px; color:var(--ink) !important; }
@media (max-width:680px){ .app-title{ font-size:2.1rem; } }

/* sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--brand-3), #f8fbff 30%, #ffffff 100%) !important;
  border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] *{ color: var(--ink) !important; }

/* inputs */
textarea, input[type="text"], input[type="search"], input[type="email"], input[type="url"]{
  background:#ffffff !important; color: var(--ink) !important;
  border:1px solid #dbe4ff !important; border-radius:10px !important;
}
[data-baseweb="textarea"] textarea{ background:#ffffff !important; color:var(--ink) !important; }

/* uploader */
[data-testid="stFileUploader"] *{ color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"]{ background: var(--muted) !important; border:2px dashed var(--brand-2) !important; }
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] svg { color: var(--ink) !important; fill: var(--ink) !important; }

/* buttons */
.stButton > button{
  background: linear-gradient(90deg, var(--brand), #60a5fa) !important;
  color:#fff !important; border:none; border-radius:10px; padding:.55rem 1rem;
  box-shadow:0 6px 16px rgba(59,130,246,.18);
}
.stButton > button:hover{ filter:brightness(1.05); }

/* progress */
[data-testid="stProgress"] > div > div{ background: linear-gradient(90deg, var(--brand), #60a5fa, #22c55e) !important; }

/* table/dataframe */
table{ border-collapse:collapse; width:100%; border-radius:12px; overflow:hidden; }
th{ background:#f3f6ff; font-weight:700; padding:10px; color:#1e3a8a !important; }
td{ padding:10px; border-top:1px solid #e5e7eb; color:var(--ink) !important; }
tr:nth-child(even) td{ background:#fafcff; }
.stDataFrame, .stTable, .stDataFrame div, .stTable div { color: var(--ink) !important; }

/* tabs */
[data-testid="stTabs"] button, [data-testid="stTabs"] button p, [data-testid="stTabs"] button span { color: var(--ink) !important; }

/* code/pre */
[data-testid="stMarkdownContainer"] pre, [data-testid="stMarkdownContainer"] code{
  background:#f8fbff !important; color:var(--ink) !important;
  border:1px solid #e6eefc !important; border-radius:8px !important;
}

/* JSON */
[data-testid="stJson"], [data-testid="stJson"] *{ color: var(--ink) !important; }
[data-testid="stJson"] pre, [data-testid="stJson"] code{
  background:#f8fbff !important; color:var(--ink) !important;
  border:1px solid #e6eefc !important; border-radius:8px !important;
}

/* hero card */
.hero{ background:#ffffffcc; backdrop-filter: blur(4px);
  border:1px solid #e6eefc; border-radius:16px; padding:18px 20px; box-shadow:var(--shadow); }
.hero *{ color: var(--ink) !important; }

/* metric pills */
.metric-grid{ display:grid; gap:14px; grid-template-columns: repeat(3,1fr); margin:14px 0 2px; }
.metric{ background:var(--bg); border:1px solid #e6eefc; border-radius:14px; padding:14px; box-shadow:var(--shadow); }
.metric .k{ font-size:.82rem; color:var(--ink-2) !important; }
.metric .v{ font-size:1.35rem; font-weight:800; margin-top:2px; color:var(--ink) !important; }

/* chips */
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px; }
.chip{ background:#e8f0ff; border:1px solid #cfe0ff; color:#1e3a8a !important;
  border-radius:999px; padding:.28rem .6rem; font-weight:600; font-size:.85rem; }

/* small */
.small{ color:#64748b !important; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar: JD ----------------
with st.sidebar:
    st.header("🔎 Job Description")
    jd_text = st.text_area("Paste JD to evaluate fit", height=180, placeholder="Tech stack, responsibilities, required experience…")
    jd_weight_skills = st.slider("Skill weight", 0.0, 1.0, 0.6, 0.05)
    jd_weight_text = 1.0 - jd_weight_skills
    st.caption("Fit = Skill overlap × %.0f%% + Semantic similarity × %.0f%%" % (jd_weight_skills*100, jd_weight_text*100))

# ---------------- OCR + Extraction ----------------
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def has_tesseract():
    return shutil.which("tesseract") is not None

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_image_bytes(file_bytes: bytes):
    # Try Tesseract first (if present)
    if has_tesseract():
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            txt = pytesseract.image_to_string(img)
            if txt.strip():
                return txt, None
        except Exception:
            pass
    # Fallback to EasyOCR
    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        reader = get_easyocr_reader()
        result = reader.readtext(np.array(img), detail=0, paragraph=True)
        return "\n".join(result), "Used EasyOCR fallback"
    except Exception as e:
        return "", f"OCR error: {e}"

def extract_text_from_pdf_bytes(file_bytes: bytes):
    # Try text layer first
    with BytesIO(file_bytes) as f:
        try:
            text = pdf_extract_text(f) or ""
        except Exception:
            text = ""
    if text and len(text.strip()) > 20:
        return text, None
    # Scanned? -> OCR via PyMuPDF render + EasyOCR
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        reader = get_easyocr_reader()
        chunks = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            ocr_text = reader.readtext(np.array(img), detail=0, paragraph=True)
            if ocr_text:
                chunks.append("\n".join(ocr_text))
        doc.close()
        joined = "\n\n".join(chunks)
        if joined.strip():
            return joined, "Used EasyOCR (scanned PDF)"
        return "", "No text found (scan quality too low)"
    except Exception as e:
        return "", f"OCR error: {e}"

def extract_text_smart_bytes(file_bytes: bytes, filename: str):
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    if fname.endswith((".docx", ".doc")):
        return extract_text_from_docx_bytes(file_bytes), None
    if fname.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return extract_text_from_image_bytes(file_bytes)
    try:
        return file_bytes.decode("utf-8"), None
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore"), None

# ---------------- Regexes ----------------
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(?:\+?\d{1,3}[\s-]?)?\d{10}')
LINK_RE  = re.compile(r'(https?://\S+|linkedin\.com/\S+|github\.com/\S+)')

# ---------------- Skill Ontology ----------------
SKILL_ONTOLOGY = {
    "python": ["python", "py", "numpy", "pandas", "scikit-learn"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c": ["c language"],
    "javascript": ["javascript", "js", "ecmascript"],
    "react": ["react", "react.js", "reactjs"],
    "node": ["node", "node.js", "nodejs", "express"],
    "django": ["django"],
    "flask": ["flask"],
    "sql": ["sql", "postgresql", "mysql", "sqlite"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "ec2", "s3", "lambda", "cloudwatch"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "pandas": ["pandas", "dataframe", "data wrangling"],
    "numpy": ["numpy", "ndarray", "vectorized"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf", "keras"],
    "git": ["git", "version control"],
    "excel": ["excel", "spreadsheets"],
    # soft
    "communication": ["communication", "stakeholder management", "presentation"],
    "leadership": ["leadership", "mentoring", "team lead"],
    "problem-solving": ["problem solving", "critical thinking", "analytical"],
}

# ---------------- Timeline Parsing ----------------
DATE_RANGE_RE = re.compile(
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s?\d{4}|\d{4})\s*(?:to|-|–|—)\s*(Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s?\d{4}|\d{4})',
    re.IGNORECASE
)

def detect_timeline(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    events = []
    for i, ln in enumerate(lines):
        m = DATE_RANGE_RE.search(ln)
        if not m:
            continue
        s_raw, e_raw = m.group(1), m.group(2)
        s = dateparser.parse(s_raw)
        e = datetime.now() if (e_raw and e_raw.lower() == "present") else dateparser.parse(e_raw)
        ctx = " ".join(lines[max(0, i-1): min(len(lines), i+2)])
        parts = re.split(r'[,|–—-]+', ctx)
        role = parts[0].strip()[:60] if parts else ""
        company = parts[1].strip()[:60] if len(parts) > 1 else ""
        if s and e:
            events.append({
                "start_raw": s_raw, "end_raw": e_raw,
                "start": s.isoformat(), "end": e.isoformat(),
                "role": role, "company": company
            })
    return events

# ---------------- Models ----------------
@st.cache_resource
def load_models():
    # grouped_entities=True for cleaner labels; also normalize later
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return ner, sent_model

def norm_ent_label(e: dict) -> str:
    lab = (e.get("entity_group") or e.get("entity") or "").upper()
    if lab.startswith(("B-","I-")):
        lab = lab.split("-", 1)[1]
    return {"PER":"PERSON","LOC":"LOCATION","ORG":"ORG","MISC":"MISC"}.get(lab, lab or "ENTITY")

def run_ner(ner_model, text: str):
    if not text.strip():
        return []
    ents = ner_model(text)
    out = []
    for e in ents:
        out.append({
            "word": e.get("word", "").strip(),
            "label": norm_ent_label(e),
            "score": float(e.get("score", 0.0))
        })
    return out

# ---------------- Summary (extractive) ----------------
def extractive_summary(sent_model, text: str, top_k=3):
    sents = [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]
    if not sents:
        return ""
    embs = sent_model.encode(sents, convert_to_tensor=True)
    doc_emb = embs.mean(dim=0)
    cos = util.cos_sim(doc_emb, embs)[0]
    idxs = cos.argsort(descending=True)[:top_k].tolist()
    return " ".join(sents[i] for i in sorted(idxs))

# ---------------- Embedding-based Skill Extraction ----------------
def candidate_phrases(text: str):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\.#-]+", text)
    phrases = set()
    for i in range(len(tokens)):
        phrases.add(tokens[i].lower())
        if i + 1 < len(tokens):
            phrases.add((tokens[i] + " " + tokens[i + 1]).lower())
        if i + 2 < len(tokens):
            tri = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}".lower()
            if len(tri.split()) <= 3:
                phrases.add(tri)
    phrases = [p for p in phrases if 2 <= len(p) <= 40]
    return list(phrases)[:2000]

def embed_skills(sent_model):
    all_terms, id_map = [], []
    for c, syns in SKILL_ONTOLOGY.items():
        for s in [c] + syns:
            all_terms.append(s)
            id_map.append(c)
    embs = sent_model.encode(all_terms, convert_to_tensor=True, normalize_embeddings=True)
    return all_terms, id_map, embs

def extract_skills_embedding(sent_model, text: str, top_k=25, sim_thresh=0.45):
    phrases = candidate_phrases(text)
    if not phrases:
        return {"hard": [], "soft": [], "all": []}, {}
    phr_embs = sent_model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True)
    all_terms, id_map, ont_embs = embed_skills(sent_model)
    sim = util.cos_sim(phr_embs, ont_embs)  # (P x T)
    matched = {}
    for i, phr in enumerate(phrases):
        j = int(sim[i].argmax())
        score = float(sim[i][j])
        if score >= sim_thresh:
            canon = id_map[j]
            prev = matched.get(canon, {"score": 0.0, "phr": phr})
            if score > prev["score"]:
                matched[canon] = {"score": score, "phr": phr}
    ranked = sorted(matched.items(), key=lambda x: x[1]["score"], reverse=True)
    picked = [k for k, _ in ranked[:top_k]]
    details = {k: v for k, v in matched.items()}
    hard = [s for s in picked if s not in {"communication", "leadership", "problem-solving"}]
    soft = [s for s in picked if s in {"communication", "leadership", "problem-solving"}]
    return {"hard": hard, "soft": soft, "all": picked}, details

# ---------------- Structured Snapshot & Suggestions ----------------
def structured_snapshot(sent_model, raw_text: str, top_k=3):
    sents = [s.strip() for s in nltk.tokenize.sent_tokenize(raw_text) if s.strip()]
    if not sents:
        return {}
    cats = {
        "Achievements": "Quantified achievements and impact.",
        "Responsibilities": "Day-to-day duties.",
        "Tech & Tools": "Technologies, frameworks, tools.",
        "Leadership": "Leadership, mentoring, ownership.",
        "Education & Certs": "Education and certifications."
    }
    emb_sents = sent_model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    emb_cats  = sent_model.encode(list(cats.values()), convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb_cats, emb_sents)  # (C x S)
    taken = set()
    snapshot = {k: [] for k in cats.keys()}
    for ci, cname in enumerate(cats.keys()):
        order = sim[ci].argsort(descending=True).tolist()
        picks = []
        for idx in order:
            if len(picks) >= top_k:
                break
            if idx in taken:
                continue
            txt = sents[idx]
            if len(txt) > 220:
                continue
            picks.append(txt)
            taken.add(idx)
        snapshot[cname] = picks
    return snapshot

def render_list(label, items):
    st.markdown(f"**{label}:**")
    if not items:
        st.markdown("—")
        return
    st.markdown(
        "<div class='chips'>" +
        "".join([f"<span class='chip'>{html.escape(str(x))}</span>" for x in items]) +
        "</div>", unsafe_allow_html=True
    )

def detect_gaps(timeline, min_months=6):
    gaps = []
    if not timeline or len(timeline) < 2:
        return gaps
    items = sorted(timeline, key=lambda r: r["start"])
    for i in range(1, len(items)):
        prev_end = pd.to_datetime(items[i - 1]["end"])
        cur_start = pd.to_datetime(items[i]["start"])
        if cur_start > prev_end:
            diff = relativedelta(cur_start, prev_end)
            months = diff.years * 12 + diff.months
            if months >= min_months:
                gaps.append({
                    "after": items[i - 1].get("role") or items[i - 1].get("company"),
                    "before": items[i].get("role") or items[i].get("company"),
                    "months": months
                })
    return gaps

def make_specific_suggestions(text, skills_all, jd_text_clean, jd_skills, missing_skills, timeline):
    tips = []
    # JD-driven
    if jd_text_clean and jd_skills:
        if missing_skills:
            tips.append("Missing JD skills: " + ", ".join(missing_skills[:3]) + ". Add bullets with real usage.")
        else:
            tips.append("Good coverage. Mirror JD nouns/responsibilities to boost ATS match.")
    # Quantification
    if sum(ch.isdigit() for ch in text) < 10:
        ex = (skills_all[0] if skills_all else "the pipeline")
        tips.append(f"Quantify 2–3 bullets (impact/scale). E.g., “Improved {ex} to cut runtime 35% for 120k rows.”")
    # Gaps
    for g in detect_gaps(timeline)[:2]:
        tips.append(f"Explain a {g['months']}-month gap between “{g['after']}” and “{g['before']}”.")
    # Portfolio
    if any(s in skills_all for s in ["python", "javascript"]) and jd_text_clean:
        tips.append("Add a GitHub/portfolio link with 1–2 JD-aligned repos.")
    # Section health
    if "aws" in jd_skills and "aws" not in skills_all:
        tips.append("If you’ve used AWS, list exact services (EC2, S3, Lambda).")
    if "sql" in jd_skills and "sql" not in skills_all:
        tips.append("Add a SQL bullet (joins/window functions/optimization) tied to a dataset.")
    # Soft skills
    if not any(s in skills_all for s in ["communication", "leadership", "problem-solving"]):
        tips.append("Add one soft-skill bullet tied to outcome (e.g., led 3 interns to deliver X).")
    # Length
    word_count = len(re.findall(r"\w+", text))
    if word_count > 1200:
        tips.append("Condense to 1–2 pages; prioritize last 3–4 years.")
    # dedupe
    out, seen = [], set()
    for t in tips:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out[:8]

# ---------------- Main UI ----------------
st.markdown("<h1 class='app-title'>📑 AI Resume Parser</h1>", unsafe_allow_html=True)
st.markdown('<span class="header-badge">AI-powered • BERT NER • Semantic Matching</span>', unsafe_allow_html=True)
st.caption("Transformer-powered parsing: embedding-based skills, JD fit with hire probability, and a visual timeline.")

st.markdown("<div style='color:var(--ink); font-weight:600;'>Upload a resume</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(" ", type=["pdf","docx","doc","png","jpg","jpeg","bmp","tiff"])
anonymize_flag = st.checkbox("Anonymize PII", value=False)

if uploaded and st.button("🚀 Parse Resume"):
    raw = uploaded.read()
    filename = uploaded.name
    text, ocr_notice = extract_text_smart_bytes(raw, filename)
    if ocr_notice:
        st.warning(ocr_notice)

    if not text or len(text.strip()) < 10:
        st.error("Could not extract text. If scanned PDF, try a clearer image/PDF.")
    else:
        # Load models
        ner_model, sent_model = load_models()

        # NER
        ner_results = run_ner(ner_model, text[:20000])

        # Contacts
        emails = EMAIL_RE.findall(text)
        phones = PHONE_RE.findall(text)
        links = LINK_RE.findall(text)

        # Name guess
        name_guess = ""
        for ent in ner_results:
            if ent["label"] == "PERSON":
                name_guess = ent["word"]; break
        if not name_guess and emails:
            name_guess = emails[0].split("@")[0]

        # Anonymize
        if anonymize_flag:
            name_guess = "CANDIDATE_XXXX"
            emails = ["hidden@example.com"] if emails else []
            phones = ["hidden"] if phones else []

        # Summary + Skills
        summary = extractive_summary(sent_model, text)
        skills_ml, skill_details = extract_skills_embedding(sent_model, text, top_k=25, sim_thresh=0.45)
        skills_all = sorted(set(skills_ml["all"]))

        # Resume score (simple heuristic)
        resume_score = min(100, 35 + len(skills_all)*2 + (10 if emails else 0) + (10 if phones else 0))

        # Timeline
        timeline = detect_timeline(text)

        # JD fit + hire probability
        jd_text_clean = (jd_text or "").strip()
        jd_skills = []
        if jd_text_clean:
            jd_skills = extract_skills_embedding(sent_model, jd_text_clean, top_k=20, sim_thresh=0.45)["all"]
        missing_skills = [s for s in jd_skills if s not in skills_all]
        sim = 0.0
        if jd_text_clean:
            cand_profile = " ".join([" ".join(skills_all), summary, text[:1200]])
            emb_cand = sent_model.encode(cand_profile, convert_to_tensor=True, normalize_embeddings=True)
            emb_jd   = sent_model.encode(jd_text_clean, convert_to_tensor=True, normalize_embeddings=True)
            sim = float(util.cos_sim(emb_cand, emb_jd).item())
        skill_overlap = 1.0 - (len(missing_skills)/len(jd_skills)) if jd_skills else 0.0
        fit_score = (jd_weight_skills * skill_overlap + jd_weight_text * sim)
        fit_score_pct = int(round(100 * fit_score))
        hire_prob = 1.0 / (1.0 + math.exp(-6*(fit_score - 0.55)))
        hire_prob_pct = int(round(100 * hire_prob))
        if not jd_text_clean:
            suggestion = "ℹ️ Add a JD to get a fit suggestion."
        elif hire_prob >= 0.75 and len(missing_skills) <= 2:
            suggestion = "✅ Strong fit — proceed to interview."
        elif hire_prob >= 0.55:
            suggestion = "🟡 Borderline — consider technical screen."
        else:
            suggestion = "❌ Low fit — keep in pipeline."

        # ---- Hero Card ----
        st.markdown(f"""
<div class="hero">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="font-size:28px;">🗂️</div>
    <div>
      <h2 style="margin:0; line-height:1.15; color:var(--ink) !important;">{name_guess or "Candidate"}</h2>
      <div class="small">📊 Resume Score: <b style="color:var(--ink) !important;">{resume_score}/100</b></div>
    </div>
  </div>
  <div style="margin-top:10px; color:var(--ink) !important;">💡 {("Skilled in " + ", ".join(skills_all[:6])) if skills_all else "—"}</div>
</div>
""", unsafe_allow_html=True)

        # JD fit box + progress
        st.markdown(f"""
<div style="margin-top:0.5rem; padding:1rem; background:#f6faff; border:1px solid #e6f0ff; border-radius:12px; color:var(--ink) !important;">
  <b>JD Fit:</b> {fit_score_pct}/100 &nbsp;•&nbsp; <b>Hire probability:</b> {hire_prob_pct}%<br/>
  {suggestion}<br/>
  <span class="small">Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}</span>
</div>
""", unsafe_allow_html=True)
        st.progress(fit_score_pct)

        # Metric pills
        st.markdown(f"""
<div class="metric-grid">
  <div class="metric"><div class="k">Skills (ML)</div><div class="v">{len(skills_all)}</div></div>
  <div class="metric"><div class="k">Emails</div><div class="v">{len(emails) if emails else 0}</div></div>
  <div class="metric"><div class="k">Links</div><div class="v">{len(links) if links else 0}</div></div>
</div>
""", unsafe_allow_html=True)

        # ---- Tabs ----
        tabs = st.tabs(["🏠 Overview","🧩 Skills","🔎 Entities","📈 Timeline","🧾 JSON"])

        with tabs[0]:
            st.subheader("Overview")
            render_list("Emails", emails)
            render_list("Phones", phones)
            render_list("Links", links)

            st.markdown("### 🔎 Structured Snapshot")
            snap = structured_snapshot(sent_model, text, top_k=3)
            if snap:
                cols = st.columns(2)
                left = ["Achievements","Responsibilities","Tech & Tools"]
                right = ["Leadership","Education & Certs"]
                with cols[0]:
                    for k in left:
                        bullets = snap.get(k, [])
                        if bullets:
                            st.markdown(f"**{k}**")
                            for b in bullets:
                                st.markdown(f"- {b}")
                with cols[1]:
                    for k in right:
                        bullets = snap.get(k, [])
                        if bullets:
                            st.markdown(f"**{k}**")
                            for b in bullets:
                                st.markdown(f"- {b}")
            else:
                st.success("**Summary:** " + (summary or "—"))

            st.markdown("### ✨ Resume Improvement Suggestions")
            st.markdown("<div class='small'>Tailored to this resume and JD</div>", unsafe_allow_html=True)
            specific = make_specific_suggestions(text, skills_all, jd_text_clean, jd_skills, missing_skills, timeline)
            for s in specific:
                st.markdown(f"- {s}")

        with tabs[1]:
            st.subheader("Skills")
            st.markdown("**Hard skills**")
            render_list("", skills_ml["hard"])
            st.markdown("**Soft skills**")
            render_list("", skills_ml["soft"])

            if jd_text_clean and jd_skills:
                st.markdown("**JD vs Resume — skill diff**")
                diff_df = pd.DataFrame({
                    "JD Skill": jd_skills,
                    "Covered in Resume": [s in skills_all for s in jd_skills]
                })
                st.dataframe(diff_df)

        with tabs[2]:
            st.subheader("Entities (BERT NER)")
            if ner_results:
                df_ents = pd.DataFrame(ner_results)
                df_ents = df_ents.rename(columns={"word":"Text","label":"Entity","score":"Confidence"})
                df_ents["Confidence"] = df_ents["Confidence"].map(lambda x: round(float(x), 2))
                st.dataframe(df_ents)
            else:
                st.write("No entities detected.")

        with tabs[3]:
            st.subheader("Career Timeline")
            if timeline:
                df_t = pd.DataFrame(timeline)
                try:
                    df_t["start_dt"] = pd.to_datetime(df_t["start"])
                    df_t["end_dt"]   = pd.to_datetime(df_t["end"])
                    df_t["label"] = df_t.apply(
                        lambda r: (r["role"] or r["company"] or r["start_raw"])[:40], axis=1
                    )
                    chart = alt.Chart(df_t).mark_bar(size=18, cornerRadius=6).encode(
                        x=alt.X('start_dt:T', title='Start'),
                        x2=alt.X2('end_dt:T', title='End'),
                        y=alt.Y('label:N', sort=None, title='Role / Company'),
                        tooltip=['role','company','start_raw','end_raw']
                    ).properties(
                        height=max(220, 42 * len(df_t)),
                        background='#ffffff'
                    ).configure_axis(
                        labelColor='#334155',
                        titleColor='#334155',
                        gridColor='#e5e7eb'
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write("Timeline table:", df_t)
            else:
                st.info("No timeline detected (need ranges like 'Jun 2022 - Dec 2022').")

        with tabs[4]:
            st.subheader("JSON Export")
            parsed = {
                "filename": filename,
                "name_guess": name_guess,
                "emails": emails,
                "phones": phones,
                "links": links,
                "skills_ml": skills_ml,
                "skills_all": skills_all,
                "summary": summary,
                "resume_score": resume_score,
                "ner_entities": ner_results,
                "timeline": timeline,
                "jd": {
                    "text_present": bool(jd_text_clean),
                    "jd_skills": jd_skills,
                    "missing_skills": missing_skills,
                    "semantic_similarity": round(sim, 3),
                    "fit_score_pct": fit_score_pct,
                    "hire_probability_pct": hire_prob_pct,
                    "suggestion": suggestion
                }
            }
            pretty_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            st.code(pretty_json, language="json")
            st.download_button(
                "Download JSON",
                data=pretty_json.encode("utf-8"),
                file_name="parsed_resume.json",
                mime="application/json"
            )
