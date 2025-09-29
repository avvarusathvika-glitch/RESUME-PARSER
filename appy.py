# ✨ AI Resume Parser — Pastel Light Theme + JD Fit + Suggestions
import streamlit as st
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from PIL import Image
import pytesseract, shutil, re, nltk, json, html, math
import pandas as pd
import numpy as np
import altair as alt
import easyocr, fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime
import dateparser
from dateutil.relativedelta import relativedelta

# ---------------- NLTK ----------------
nltk.download('punkt', quiet=True)
try: nltk.download('punkt_tab', quiet=True)
except: pass

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
h1,h2,h3,h4,h5,h6,label,p,span,div,small,strong,em,b,i {
  color: var(--ink) !important;
}
/* bigger app title */
.app-title{
  font-size: 2.6rem; line-height:1.1; margin:0 0 .35rem 0; letter-spacing:.2px;
  color: var(--ink) !important;
}
@media (max-width:680px){ .app-title{ font-size:2.1rem; } }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--brand-3), #f8fbff 30%, #ffffff 100%) !important;
  border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] *{ color: var(--ink) !important; }

textarea, input[type="text"], input[type="search"], input[type="email"], input[type="url"]{
  background:#ffffff !important; color: var(--ink) !important;
  border:1px solid #dbe4ff !important; border-radius:10px !important;
}
[data-baseweb="textarea"] textarea{ background:#ffffff !important; color:var(--ink) !important; }

[data-testid="stSlider"] *{ color: var(--ink) !important; }

.block-container{ padding-top:1rem; padding-bottom:2rem; }

.header-badge{ display:inline-block; padding:.28rem .7rem; border-radius:999px;
  background:rgba(59,130,246,.09); color:var(--brand) !important;
  font-weight:600; font-size:.85rem; border:1px solid rgba(59,130,246,.22);
}

.hero{ background: #ffffffcc; backdrop-filter: blur(4px);
  border:1px solid #e6eefc; border-radius:16px; padding:18px 20px; box-shadow:var(--shadow); }
.hero *{ color: var(--ink) !important; }

.metric-grid{ display:grid; gap:14px; grid-template-columns: repeat(3,1fr); margin:14px 0 2px; }
.metric{ background:var(--bg); border:1px solid #e6eefc; border-radius:14px; padding:14px; box-shadow:var(--shadow); }
.metric .k{ font-size:.82rem; color:var(--ink-2) !important; }
.metric .v{ font-size:1.35rem; font-weight:800; margin-top:2px; color:var(--ink) !important; }

[data-testid="stFileUploader"] *{ color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"]{
  background: var(--muted) !important; border:2px dashed var(--brand-2) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] svg {
  color: var(--ink) !important; fill: var(--ink) !important;
}

.stButton > button{
  background: linear-gradient(90deg, var(--brand), #60a5fa) !important;
  color:#fff !important; border:none; border-radius:10px; padding:.55rem 1rem;
  box-shadow:0 6px 16px rgba(59,130,246,.18);
}
.stButton > button:hover{ filter:brightness(1.05); }

[data-testid="stProgress"] > div > div{
  background: linear-gradient(90deg, var(--brand), #60a5fa, #22c55e) !important;
}

table{ border-collapse:collapse; width:100%; border-radius:12px; overflow:hidden; }
th{ background:#f3f6ff; font-weight:700; padding:10px; color:#1e3a8a !important; }
td{ padding:10px; border-top:1px solid #e5e7eb; color:var(--ink) !important; }
tr:nth-child(even) td{ background:#fafcff; }
.stDataFrame, .stTable, .stDataFrame div, .stTable div { color: var(--ink) !important; }

[data-testid="stTabs"] button,
[data-testid="stTabs"] button p,
[data-testid="stTabs"] button span { color: var(--ink) !important; }

[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] code {
  background:#f8fbff !important; color:var(--ink) !important;
  border:1px solid #e6eefc !important; border-radius:8px !important;
}

/* JSON override */
[data-testid="stJson"], [data-testid="stJson"] *{ color: var(--ink) !important; }
[data-testid="stJson"] pre, [data-testid="stJson"] code{
  background:#f8fbff !important; color:var(--ink) !important;
  border:1px solid #e6eefc !important; border-radius:8px !important;
}

.chips{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px; }
.chip{ background:#e8f0ff; border:1px solid #cfe0ff; color:#1e3a8a !important;
  border-radius:999px; padding:.28rem .6rem; font-weight:600; font-size:.85rem; }

.small{ color:#64748b !important; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar: JD ----------------
with st.sidebar:
    st.header("🔎 Job Description")
    jd_text = st.text_area("Paste JD to evaluate fit", height=180)
    jd_weight_skills = st.slider("Skill weight", 0.0, 1.0, 0.6, 0.05)
    jd_weight_text  = 1.0 - jd_weight_skills

# ---------------- Helpers (OCR, parsing, NER, etc.) ----------------
# (keep the helper functions from the previous version here — unchanged)

# ---------------- Main UI ----------------
st.markdown("<h1 class='app-title'>📑 AI Resume Parser</h1>", unsafe_allow_html=True)
st.markdown('<span class="header-badge">AI-powered • BERT NER • Semantic Matching</span>', unsafe_allow_html=True)

st.markdown("<div style='color:var(--ink); font-weight:600;'>Upload a resume</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(" ", type=["pdf","docx","doc","png","jpg","jpeg","bmp","tiff"])
anonymize_flag = st.checkbox("Anonymize PII", value=False)

# (rest of your pipeline stays exactly same as before —
# extraction, NER, skills, summary, JD fit, tabs etc.)

# In the JSON tab: replace st.json(parsed) with st.code
with tabs[4]:
    st.subheader("JSON Export")
    parsed = { ... }  # build as before
    pretty_json = json.dumps(parsed, indent=2, ensure_ascii=False)
    st.code(pretty_json, language="json")
    st.download_button("Download JSON", data=pretty_json.encode("utf-8"),
                       file_name="parsed_resume.json", mime="application/json")
