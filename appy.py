# appy.py

import streamlit as st
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from PIL import Image
import pytesseract, shutil
import re, nltk, json
import pandas as pd
import numpy as np
import altair as alt
import easyocr, fitz
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from dateutil.relativedelta import relativedelta
import html

# ---------- NLTK setup ----------
nltk.download("punkt")
nltk.download("punkt_tab")

# ---------- Style (pastel blue light theme) ----------
st.markdown("""
<style>
:root{
  --bg:#ffffff;
  --ink:#0f172a;
  --muted:#f6f9ff;
  --card:#ffffff;
  --brand:#3b82f6;
  --brand-2:#93c5fd;
  --brand-3:#dbeafe;
  --ink-2:#334155;
  --shadow:0 10px 30px rgba(30,64,175,.06);
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% -10%, var(--brand-3) 0%, transparent 40%),
    radial-gradient(900px 500px at 110% 10%, #e0f2fe 0%, transparent 35%),
    var(--bg) !important;
  color: var(--ink) !important;
}

h1, h2, h3, h4, h5, h6, label, p, span, div, small {
  color: inherit !important;
}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--brand-3), #f8fbff 30%, #ffffff 100%) !important;
  border-right: 1px solid #e5e7eb;
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] *{ color: var(--ink) !important; }

textarea, input[type="text"], input[type="search"], input[type="email"], input[type="url"]{
  background:#ffffff !important;
  color: var(--ink) !important;
  border:1px solid #dbe4ff !important;
  border-radius:10px !important;
}

[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] code {
  background:#f8fbff !important;
  color: var(--ink) !important;
  border:1px solid #e6eefc !important;
  border-radius:8px !important;
}

.chips{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px; }
.chip{
  background:#e8f0ff; border:1px solid #cfe0ff; color:#1e3a8a;
  border-radius:999px; padding:.28rem .6rem; font-weight:600; font-size:.85rem;
}

.stButton > button{
  background: linear-gradient(90deg, var(--brand), #60a5fa) !important;
  color:#fff !important; border:none; border-radius:10px; padding:.55rem 1rem;
  box-shadow: 0 6px 16px rgba(59,130,246,.18);
}
.stButton > button:hover{ filter: brightness(1.05); }

table{ border-collapse:collapse; width:100%; border-radius:12px; overflow:hidden; }
th{ background:#f3f6ff; font-weight:700; padding:10px; color:#1e3a8a !important; }
td{ padding:10px; border-top:1px solid #e5e7eb; color: var(--ink) !important; }
tr:nth-child(even) td{ background:#fafcff; }

.small{ color:#64748b !important; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
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

@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def extract_text_from_docx_bytes(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image_bytes(file_bytes):
    if shutil.which("tesseract"):
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            return pytesseract.image_to_string(img), None
        except Exception: pass
    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        reader = get_easyocr_reader()
        result = reader.readtext(np.array(img), detail=0, paragraph=True)
        return "\n".join(result), "Used EasyOCR fallback"
    except Exception as e:
        return "", f"OCR error: {e}"

def extract_text_from_pdf_bytes(file_bytes):
    try:
        txt = pdf_extract_text(BytesIO(file_bytes)) or ""
    except Exception: txt = ""
    if txt.strip(): return txt, None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        reader = get_easyocr_reader()
        chunks=[]
        for page in doc:
            pix=page.get_pixmap(dpi=200)
            img=Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            res=reader.readtext(np.array(img), detail=0, paragraph=True)
            if res: chunks.append("\n".join(res))
        doc.close()
        joined="\n\n".join(chunks)
        if joined.strip(): return joined, "Used EasyOCR (scanned PDF)"
    except Exception as e:
        return "", f"OCR error: {e}"
    return "", "No text found"

def extract_text_smart_bytes(file_bytes, filename):
    fname = filename.lower()
    if fname.endswith(".pdf"): return extract_text_from_pdf_bytes(file_bytes)
    if fname.endswith((".docx",".doc")): return extract_text_from_docx_bytes(file_bytes), None
    if fname.endswith((".png",".jpg",".jpeg",".bmp",".tiff")): return extract_text_from_image_bytes(file_bytes)
    try: return file_bytes.decode("utf-8"), None
    except: return file_bytes.decode("latin-1",errors="ignore"), None

def structured_snapshot(sent_model, raw_text, top_k=3):
    sents=[s.strip() for s in nltk.tokenize.sent_tokenize(raw_text) if s.strip()]
    if not sents: return {}
    cats={
        "Achievements":"Quantified achievements, impact.",
        "Responsibilities":"Day-to-day duties.",
        "Tech & Tools":"Technologies, frameworks, tools.",
        "Leadership":"Leadership, mentoring, ownership.",
        "Education & Certs":"Education, degrees, certifications."
    }
    emb_sents=sent_model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    emb_cats=sent_model.encode(list(cats.values()), convert_to_tensor=True, normalize_embeddings=True)
    sim=util.cos_sim(emb_cats, emb_sents)
    taken=set(); snapshot={k:[] for k in cats.keys()}
    for ci,cname in enumerate(cats.keys()):
        order=sim[ci].argsort(descending=True).tolist()
        picks=[]
        for idx in order:
            if len(picks)>=top_k: break
            if idx in taken: continue
            txt=sents[idx]
            if len(txt)>220: continue
            picks.append(txt); taken.add(idx)
        snapshot[cname]=picks
    return snapshot

def detect_gaps(timeline, min_months=6):
    gaps=[]
    if not timeline or len(timeline)<2: return gaps
    items=sorted(timeline,key=lambda r:r["start"])
    for i in range(1,len(items)):
        prev_end=pd.to_datetime(items[i-1]["end"])
        cur_start=pd.to_datetime(items[i]["start"])
        if cur_start>prev_end:
            diff=relativedelta(cur_start, prev_end)
            months=diff.years*12+diff.months
            if months>=min_months:
                gaps.append({"after":items[i-1].get("role"),"before":items[i].get("role"),"months":months})
    return gaps

def make_specific_suggestions(text, skills_all, jd_text_clean, jd_skills, missing_skills, timeline):
    tips=[]
    if jd_text_clean and jd_skills:
        if missing_skills:
            tips.append("Missing JD skills: " + ", ".join(missing_skills[:3]) + ". Add bullets with real usage.")
    if sum(ch.isdigit() for ch in text)<10:
        tips.append("Quantify impact. E.g. 'Reduced runtime 35% for 120k rows.'")
    gaps=detect_gaps(timeline)
    for g in gaps[:2]:
        tips.append(f"Explain {g['months']} month gap between {g['after']} and {g['before']}.")
    if not any(s.lower() in ["communication","leadership","problem-solving"] for s in skills_all):
        tips.append("Add one soft-skill bullet tied to an outcome.")
    word_count=len(re.findall(r"\\w+",text))
    if word_count>1200:
        tips.append("Condense to 1–2 pages. Prioritize last 3–4 years.")
    return tips[:8]

# ---------- Models ----------
@st.cache_resource
def load_models():
    return (
        pipeline("ner", model="dslim/bert-base-NER"),
        SentenceTransformer("all-MiniLM-L6-v2")
    )
ner_model, sent_model = load_models()

# ---------- UI ----------
st.title("📑 AI Resume Parser")
st.markdown('<span class="header-badge">AI-powered • BERT NER • Semantic Matching</span>', unsafe_allow_html=True)

jd_text = st.sidebar.text_area("🔍 Job Description", "Tech stack, responsibilities, required experience…")
skill_weight = st.sidebar.slider("Skill weight", 0.0, 1.0, 0.6, 0.05)

up = st.file_uploader("Upload a resume", type=["pdf","docx","doc","png","jpg","jpeg","bmp","tiff"])
anon = st.checkbox("Anonymize PII")

if up:
    raw=up.read(); filename=up.name
    text, notice=extract_text_smart_bytes(raw, filename)
    if notice: st.warning(notice)
    if not text.strip(): st.error("No text extracted."); st.stop()

    # Fake extractors for demo
    emails=re.findall(r"[\\w\\.-]+@[\\w\\.-]+", text)
    phones=re.findall(r"\\+?\\d[\\d -]{8,}\\d", text)
    links=re.findall(r"https?://\\S+|www\\.\\S+", text)
    name_guess=(text.split("\\n")[0] if text else "Candidate")

    # Example: treat skills as top keywords
    skills_all=list(set(re.findall(r"(?i)python|sql|aws|java|tensorflow|pandas|git|communication|leadership", text)))
    jd_skills=list(set(re.findall(r"(?i)python|sql|aws|java|tensorflow|pandas|git|communication|leadership", jd_text)))
    missing_skills=[s for s in jd_skills if s not in skills_all]

    # Fit scoring
    if jd_text.strip():
        emb_resume=sent_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        emb_jd=sent_model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
        sem_sim=float(util.cos_sim(emb_resume, emb_jd).item())
        skill_overlap=len(set(skills_all)&set(jd_skills))/len(jd_skills) if jd_skills else 0
        fit=skill_weight*skill_overlap+(1-skill_weight)*sem_sim
        resume_score=int(fit*100)
    else: resume_score=50

    tabs=st.tabs(["🏠 Overview","🧩 Skills","🔎 Entities","📈 Timeline"])
    with tabs[0]:
        st.subheader("Overview")
        render_list("Emails", emails)
        render_list("Phones", phones)
        render_list("Links", links)

        st.markdown("### 🔎 Structured Snapshot")
        snap=structured_snapshot(sent_model, text, top_k=3)
        if snap:
            cols=st.columns(2)
            left=["Achievements","Responsibilities","Tech & Tools"]
            right=["Leadership","Education & Certs"]
            with cols[0]:
                for k in left:
                    if snap[k]: st.markdown(f"**{k}**"); [st.markdown(f"- {b}") for b in snap[k]]
            with cols[1]:
                for k in right:
                    if snap[k]: st.markdown(f"**{k}**"); [st.markdown(f"- {b}") for b in snap[k]]

        st.markdown("### ✨ Resume Improvement Suggestions")
        specific=make_specific_suggestions(text, skills_all, jd_text, jd_skills, missing_skills, [])
        for s in specific: st.markdown(f"- {s}")

    with tabs[1]:
        st.subheader("Skills")
        render_list("Extracted Skills", skills_all)
        if jd_skills:
            render_list("JD Skills", jd_skills)
            if missing_skills: render_list("Missing Skills", missing_skills)

    with tabs[2]:
        st.subheader("Entities")
        ents=ner_model(text)
        for e in ents[:15]:
            st.write(f"{e['word']} → {e['entity_group']} ({e['score']:.2f})")

    with tabs[3]:
        st.subheader("Timeline (demo)")
        st.info("Timeline extraction not fully implemented here.")
