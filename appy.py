# ✨ AI Resume Parser — Light Theme + Embedding Skills + JD Fit + Hire Probability + Timeline + Suggestions
import streamlit as st
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text
import docx
from PIL import Image
import pytesseract
from pytesseract import TesseractNotFoundError
import shutil
import re
import nltk
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import dateparser
import pandas as pd
import altair as alt
import json
from datetime import datetime
import math

# ---- NLTK resources (needed on Streamlit Cloud) ----
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# ---- Page + Light Theme CSS ----
st.set_page_config(layout="wide", page_title="AI Resume Parser", page_icon="📑")
st.markdown("""
<style>
/* ========= Pastel Blue Light Theme (Strict) ========= */
:root{
  --bg:#ffffff;
  --ink:#0f172a;                 /* slate-900 */
  --muted:#f6f9ff;               /* very light blue */
  --card:#ffffff;
  --brand:#3b82f6;               /* blue-500 */
  --brand-2:#93c5fd;             /* blue-300 */
  --brand-3:#dbeafe;             /* blue-100 */
  --accent:#a5b4fc;              /* indigo-300 */
  --ink-2:#334155;               /* slate-700 */
  --shadow:0 10px 30px rgba(30,64,175,.06);
}

/* App background + text */
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% -10%, var(--brand-3) 0%, transparent 40%),
    radial-gradient(900px 500px at 110% 10%, #e0f2fe 0%, transparent 35%),
    var(--bg) !important;
  color: var(--ink) !important;
}

/* Force headings to dark ink (some themes make them white) */
h1, h2, h3, h4, h5, h6, label, p, span, div, small {
  color: inherit !important;
}

/* Sidebar – soft gradient + dark text */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--brand-3), #f8fbff 30%, #ffffff 100%) !important;
  border-right: 1px solid #e5e7eb;
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] *{
  color: var(--ink) !important;
}

/* Inputs: textareas / text inputs white, not black */
textarea, input[type="text"], input[type="search"], input[type="email"], input[type="url"]{
  background:#ffffff !important;
  color: var(--ink) !important;
  border:1px solid #dbe4ff !important;
  border-radius:10px !important;
}
[data-baseweb="textarea"] textarea{ background:#ffffff !important; color:var(--ink) !important; }

/* Slider label + ticks */
[data-testid="stSlider"] *{ color: var(--ink) !important; }

/* container spacing */
.block-container{ padding-top: 1rem; padding-bottom: 2rem; }

/* header badge */
.header-badge{
  display:inline-block; padding:.28rem .7rem; border-radius:999px;
  background:rgba(59,130,246,.09); color:var(--brand); font-weight:600; font-size:.85rem;
  border:1px solid rgba(59,130,246,.22);
}

/* hero card */
.hero{
  background: linear-gradient(135deg, #ffffffcc, #ffffffcc);
  backdrop-filter: blur(4px);
  border:1px solid #e6eefc;
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: var(--shadow);
}

/* metric pills */
.metric-grid{ display:grid; gap:14px; grid-template-columns: repeat(3, minmax(0,1fr)); margin: 14px 0 2px 0; }
.metric{
  background: var(--card);
  border:1px solid #e6eefc;
  border-radius:14px; padding:14px;
  box-shadow: var(--shadow);
}
.metric .k{ font-size:.82rem; color:var(--ink-2); }
.metric .v{ font-size:1.35rem; font-weight:800; margin-top:2px; }

/* file uploader – light */
[data-testid="stFileUploaderDropzone"]{
  background: var(--muted) !important;
  border:2px dashed var(--brand-2) !important;
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] *{ color: var(--ink) !important; }
[data-testid="stFileUploader"] button{
  background:#1e293b !important; color:#fff !important; /* keep button readable */
  border-radius:8px !important; border:none !important;
}

/* primary buttons */
.stButton > button{
  background: linear-gradient(90deg, var(--brand), #60a5fa) !important;
  color:#fff !important; border:none; border-radius:10px; padding:.55rem 1rem;
  box-shadow: 0 6px 16px rgba(59,130,246,.18);
}
.stButton > button:hover{ filter: brightness(1.05); }

/* progress (fit bar) */
[data-testid="stProgress"] > div > div{
  background: linear-gradient(90deg, var(--brand), #60a5fa, #22c55e) !important;
}

/* tables */
table{ border-collapse:collapse; width:100%; border-radius:12px; overflow:hidden; }
th{ background:#f3f6ff; font-weight:700; padding:10px; color:#1e3a8a !important; }
td{ padding:10px; border-top:1px solid #e5e7eb; color: var(--ink) !important; }
tr:nth-child(even) td{ background:#fafcff; }

/* tabs */
[data-testid="stTabs"] button{ font-weight:600; color: var(--ink) !important; }

/* small text */
.small{ color:#64748b !important; font-size:.92rem; }

/* footer */
.footer{ color:#94a3b8 !important; font-size:.85rem; margin-top:18px; }
</style>
""", unsafe_allow_html=True)


# ---- Sidebar: JD input ----
with st.sidebar:
    st.header("🔎 Job Description")
    jd_text = st.text_area(
        "Paste JD to evaluate fit",
        placeholder="Tech stack, responsibilities, required experience…",
        height=180
    )
    jd_weight_skills = st.slider("Skill weight", 0.0, 1.0, 0.6, 0.05)
    jd_weight_text  = 1.0 - jd_weight_skills
    st.caption("Fit = Skill overlap × %.0f%% + Semantic similarity × %.0f%%" % (jd_weight_skills*100, jd_weight_text*100))
    st.markdown("---")
    st.caption("Tip: paste a short JD for faster, crisper results.")

# ---- Extractors ----
def extract_text_from_pdf_bytes(file_bytes):
    with BytesIO(file_bytes) as f:
        try:
            return pdf_extract_text(f)
        except Exception:
            return ""

def extract_text_from_docx_bytes(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def has_tesseract():
    return shutil.which("tesseract") is not None

def extract_text_from_image_bytes(file_bytes):
    if not has_tesseract():
        return "", "OCR unavailable (Tesseract not installed on this host)"
    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        txt = pytesseract.image_to_string(img)
        return txt, None
    except TesseractNotFoundError:
        return "", "OCR unavailable (Tesseract not installed on this host)"
    except Exception as e:
        return "", f"OCR error: {e}"

def extract_text_smart_bytes(file_bytes, filename):
    fname = filename.lower()
    if fname.endswith(".pdf"):
        txt = extract_text_from_pdf_bytes(file_bytes)
        return txt, None
    if fname.endswith((".docx",".doc")):
        return extract_text_from_docx_bytes(file_bytes), None
    if fname.endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
        return extract_text_from_image_bytes(file_bytes)  # (text, notice)
    try:
        return file_bytes.decode("utf-8"), None
    except:
        return file_bytes.decode("latin-1", errors="ignore"), None

# ---- Regex & helpers ----
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s-]?)?\d{10}')
LINK_RE  = re.compile(r'(https?://\S+|linkedin\.com/\S+|github\.com/\S+)')

# (compact skills ontology with synonyms)
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

# Date ranges like "Jun 2022 - Dec 2023" or "2019 - 2021"
DATE_RANGE_RE = re.compile(
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s?\d{4}|\d{4})\s*(?:to|-|–|—)\s*(Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s?\d{4}|\d{4})',
    re.IGNORECASE
)

def detect_timeline(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    events = []
    for i, ln in enumerate(lines):
        m = DATE_RANGE_RE.search(ln)
        if not m: continue
        s_raw, e_raw = m.group(1), m.group(2)
        s = dateparser.parse(s_raw)
        e = datetime.now() if (e_raw and e_raw.lower() == "present") else dateparser.parse(e_raw)
        context = " ".join(lines[max(0, i-1): min(len(lines), i+2)])
        parts = re.split(r'[,|–—-]+', context)
        role = parts[0].strip()[:60] if parts else ""
        company = parts[1].strip()[:60] if len(parts) > 1 else ""
        if s and e:
            events.append({
                "start_raw": s_raw, "end_raw": e_raw,
                "start": s.isoformat(), "end": e.isoformat(),
                "role": role, "company": company
            })
    return events

@st.cache_resource
def load_models():
    # For a resume-specific NER later, swap the model id here.
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return ner, sent_model

def run_ner(ner_model, text):
    if not text.strip(): return []
    ents = ner_model(text)
    return [{"entity": e.get("entity_group"), "word": e.get("word"), "score": float(e["score"])} for e in ents]

def extractive_summary(sent_model, text, top_k=3):
    sents = nltk.tokenize.sent_tokenize(text)
    if not sents: return ""
    embs = sent_model.encode(sents, convert_to_tensor=True)
    doc_emb = embs.mean(dim=0)
    cos = util.cos_sim(doc_emb, embs)[0]
    idxs = cos.argsort(descending=True)[:top_k].tolist()
    return " ".join(sents[i] for i in sorted(idxs))

# ---- ML Skill Extraction (embedding-based) ----
def candidate_phrases(text, max_len=6):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\.#-]+", text)
    phrases = set()
    for i in range(len(tokens)):
        phrases.add(tokens[i].lower())
        if i+1 < len(tokens):
            phrases.add((tokens[i] + " " + tokens[i+1]).lower())
        if i+2 < len(tokens):
            tri = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}".lower()
            if len(tri.split()) <= 3:
                phrases.add(tri)
    phrases = [p for p in phrases if 2 <= len(p) <= 40]
    return list(phrases)[:2000]

def embed_skills(sent_model):
    canon = list(SKILL_ONTOLOGY.keys())
    all_terms, id_map = [], []
    for c, syns in SKILL_ONTOLOGY.items():
        for s in [c] + syns:
            all_terms.append(s)
            id_map.append(c)
    embs = sent_model.encode(all_terms, convert_to_tensor=True, normalize_embeddings=True)
    return all_terms, id_map, embs

def extract_skills_embedding(sent_model, text, top_k=25, sim_thresh=0.45):
    phrases = candidate_phrases(text)
    if not phrases: return {"hard":[], "soft":[], "all":[]}, {}
    phr_embs = sent_model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True)
    all_terms, id_map, ont_embs = embed_skills(sent_model)
    sim = util.cos_sim(phr_embs, ont_embs)  # (P x T)
    matched = {}
    for i, phr in enumerate(phrases):
        j = int(sim[i].argmax())
        score = float(sim[i][j])
        if score >= sim_thresh:
            canon = id_map[j]
            prev = matched.get(canon, {"score":0.0, "phr":phr})
            if score > prev["score"]:
                matched[canon] = {"score": score, "phr": phr}
    ranked = sorted(matched.items(), key=lambda x: x[1]["score"], reverse=True)
    picked = [k for k,_ in ranked[:top_k]]
    details = {k:v for k,v in matched.items()}
    hard = [s for s in picked if s not in {"communication","leadership","problem-solving"}]
    soft = [s for s in picked if s in {"communication","leadership","problem-solving"}]
    return {"hard": hard, "soft": soft, "all": picked}, details

def improvement_suggestions(emails, phones, links, skills_all, summary, jd_text_clean, jd_skills, missing):
    tips = []
    if not emails: tips.append("Add a professional email at the top.")
    if not phones: tips.append("Include a phone number with country code.")
    if not links: tips.append("Add LinkedIn/GitHub/portfolio links.")
    if not summary: tips.append("Add a 2–3 line summary with role, YOE, and top skills.")
    elif len(summary) < 140: tips.append("Expand summary with one quantified achievement.")
    if len(skills_all) < 6: tips.append("List 6–10 relevant skills to pass ATS screening.")
    if missing: tips.append("Address JD gaps: " + ", ".join(missing) + " (add only if real).")
    tips.append("Quantify results (e.g., 'reduced latency by 30%', 'handled 100k users').")
    tips.append("Start bullets with verbs (Built, Led, Optimized, Automated).")
    tips.append("Use consistent dates (e.g., 'Jun 2022 – Dec 2023'). Export as PDF.")
    if jd_text_clean: tips.append("Mirror key JD terms to improve ATS match.")
    out, seen = [], set()
    for t in tips:
        if t not in seen: out.append(t); seen.add(t)
    return out[:8]

# ---- UI ----
st.title("📑 AI Resume Parser")
st.markdown('<span class="header-badge">AI-powered • BERT NER • Semantic Matching</span>', unsafe_allow_html=True)
st.caption("Transformer-powered parsing: embedding-based skills, JD fit with hire probability, and a visual timeline.")

uploaded = st.file_uploader("Upload a resume", type=["pdf","docx","doc","png","jpg","jpeg","bmp","tiff"])
anonymize_flag = st.checkbox("Anonymize PII", value=False)

if uploaded and st.button("🚀 Parse Resume"):
    raw = uploaded.read()
    filename = uploaded.name
    text, ocr_notice = extract_text_smart_bytes(raw, filename)

    if ocr_notice:
        st.warning(ocr_notice + " — upload a text-based PDF/DOCX, or ask to enable EasyOCR.")

    if not text or len(text.strip()) < 10:
        st.error("Could not extract text. If scanned PDF, try uploading as image (PNG/JPG) or enable OCR later.")
    else:
        # models
        ner_model, sent_model = load_models()
        ner_results = run_ner(ner_model, text[:20000])

        # summary
        summary = extractive_summary(sent_model, text)

        # contacts
        emails = EMAIL_RE.findall(text)
        phones = PHONE_RE.findall(text)
        links  = LINK_RE.findall(text)

        # name guess
        name_guess = ""
        for e in ner_results:
            if e["entity"] and e["entity"].lower()=="per":
                name_guess = e["word"]; break
        if not name_guess and emails:
            name_guess = emails[0].split("@")[0]

        # anonymize if chosen
        if anonymize_flag:
            name_guess = "CANDIDATE_XXXX"
            emails = ["hidden@example.com"] if emails else []
            phones = ["hidden"] if phones else []

        # ML skill extraction
        skills_ml, skill_details = extract_skills_embedding(sent_model, text, top_k=25, sim_thresh=0.45)
        skills_all = sorted(set(skills_ml["all"]))

        # resume score (simple but data-driven)
        resume_score = min(100, 35 + len(skills_all)*2 + (10 if emails else 0) + (10 if phones else 0))

        # timeline
        timeline = detect_timeline(text)

        # JD fit (skill overlap + semantic similarity)
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

        # Smooth "hire probability" (logistic on blended score scaled)
        hire_prob = 1.0 / (1.0 + math.exp(-6*(fit_score - 0.55)))  # center ~0.55
        hire_prob_pct = int(round(100*hire_prob))

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
      <h2 style="margin:0; line-height:1.15;">{name_guess or "Candidate"}</h2>
      <div class="small">📊 Resume Score: <b>{resume_score}/100</b></div>
    </div>
  </div>
  <div style="margin-top:10px;">💡 {("Skilled in " + ", ".join(skills_all[:6])) if skills_all else "—"}</div>
</div>
""", unsafe_allow_html=True)

        # JD fit box + progress
        st.markdown(f"""
<div style="margin-top:0.5rem; padding:1rem; background:#f6faff; border:1px solid #e6f0ff; border-radius:12px;">
  <b>JD Fit:</b> {fit_score_pct}/100 &nbsp;•&nbsp; <b>Hire probability:</b> {hire_prob_pct}%<br/>
  {suggestion}<br/>
  <span class="small">Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}</span>
</div>
""", unsafe_allow_html=True)
        st.progress(fit_score_pct)

        # Metric pills
        st.markdown(f"""
<div class="metric-grid">
  <div class="metric">
    <div class="k">Skills (ML)</div>
    <div class="v">{len(skills_all)}</div>
  </div>
  <div class="metric">
    <div class="k">Emails</div>
    <div class="v">{len(emails) if emails else 0}</div>
  </div>
  <div class="metric">
    <div class="k">Links</div>
    <div class="v">{len(links) if links else 0}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ---- Tabs ----
        tabs = st.tabs(["🏠 Overview","🧩 Skills","🔎 Entities","📈 Timeline","🧾 JSON"])

        with tabs[0]:
            st.subheader("Overview")
            st.write("**Emails:**", emails or "—")
            st.write("**Phones:**", phones or "—")
            st.write("**Links:**", links or "—")
            st.success("**Summary:** " + (summary or "—"))
            st.markdown("### ✨ Resume Improvement Suggestions")
            st.markdown("<div class='small'>Actionable, ATS-friendly tips</div>", unsafe_allow_html=True)
            for s in improvement_suggestions(emails, phones, links, skills_all, summary, jd_text_clean, jd_skills, missing_skills):
                st.markdown(f"- {s}")

        with tabs[1]:
            st.subheader("Skills (ML Extracted)")
            colA, colB = st.columns(2)
            colA.markdown("**Hard skills**")
            colA.write(skills_ml["hard"] or "—")
            colB.markdown("**Soft skills**")
            colB.write(skills_ml["soft"] or "—")

            st.markdown("**Top matched phrases → canonical skill (confidence)**")
            if skill_details:
                rows = []
                for canon, info in skill_details.items():
                    rows.append({"Canonical": canon, "Phrase": info["phr"], "Confidence": round(info["score"], 3)})
                st.dataframe(pd.DataFrame(rows).sort_values("Confidence", ascending=False))
            else:
                st.write("—")

        with tabs[2]:
            st.subheader("NER Entities (BERT)")
            if ner_results:
                st.dataframe(pd.DataFrame(ner_results))
            else:
                st.write("No entities detected.")

        with tabs[3]:
            st.subheader("Career Timeline")
            timeline = timeline or detect_timeline(text)
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
            st.json(parsed)
            st.download_button(
                "Download JSON",
                data=json.dumps(parsed, indent=2),
                file_name="parsed_resume.json",
                mime="application/json"
            )

# cute footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit, BERT, and Sentence-BERT.</div>", unsafe_allow_html=True)
