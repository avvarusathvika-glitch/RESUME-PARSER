# ✨ AI Resume Parser — Polished + JD Fit + Timeline
import streamlit as st
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text
import docx
from PIL import Image
import pytesseract
import re
import nltk
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import dateparser
import pandas as pd
import altair as alt
import json
from datetime import datetime
from dateutil import parser as dateparser2

# ---- NLTK resources (needed on Streamlit Cloud) ----
nltk.download('punkt', quiet=True)
# newer NLTK needs this too on some envs
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# ---- Page + Theme ----
st.set_page_config(layout="wide", page_title="AI Resume Parser", page_icon="📑")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
/* Metric cards */
[data-testid="stMetric"] {
  background: #f9fafb; padding: 1rem; border-radius: 1rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
/* Hero card */
.hero {
  background:#fdfdfd; padding:1.25rem; border-radius:1rem;
  border:1px solid #eef2f7; box-shadow:0 2px 8px rgba(0,0,0,0.05);
}
.badge {
  display:inline-block; padding:.25rem .5rem; border-radius:.5rem;
  background:#eef6ff; color:#174ea6; font-weight:600; font-size:.85rem;
}
table {border-collapse:collapse; width:100%; border-radius:8px; overflow:hidden;}
th {background:#f0f2f6; font-weight:600; padding:8px;}
td {padding:8px; border-top:1px solid #e5e7eb;}
tr:nth-child(even) td {background:#fafafa;}
.small {color:#6b7280; font-size:.9rem;}
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

# ---- Extractors ----
def extract_text_from_pdf_bytes(file_bytes):
    with BytesIO(file_bytes) as f:
        try: return pdf_extract_text(f)
        except Exception: return ""

def extract_text_from_docx_bytes(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_image_bytes(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return pytesseract.image_to_string(img)

def extract_text_smart_bytes(file_bytes, filename):
    fname = filename.lower()
    if fname.endswith(".pdf"): return extract_text_from_pdf_bytes(file_bytes)
    if fname.endswith((".docx",".doc")): return extract_text_from_docx_bytes(file_bytes)
    if fname.endswith((".png",".jpg",".jpeg",".bmp",".tiff")): return extract_text_from_image_bytes(file_bytes)
    try: return file_bytes.decode("utf-8")
    except: return file_bytes.decode("latin-1", errors="ignore")

# ---- Regex & helpers ----
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s-]?)?\d{10}')
LINK_RE  = re.compile(r'(https?://\S+|linkedin\.com/\S+|github\.com/\S+)')

CURATED_SKILLS = [
    "python","java","c++","c","javascript","react","angular","node","django","flask",
    "tensorflow","pytorch","keras","sql","postgresql","mysql","mongodb","docker",
    "kubernetes","aws","azure","gcp","pandas","numpy","scikit-learn","excel","git","communication"
]

# Date ranges like "Jun 2022 - Dec 2023" or "2019-2021"
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
        # infer role/company from neighbor lines
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

def group_skills(skills):
    tech = [s for s in skills if s in {"python","sql","tensorflow","docker","aws","git","react","django","pandas","numpy"}]
    return {"technical": tech, "non_technical": [s for s in skills if s not in tech]}

def elevator_pitch(name, skills, role):
    return f"{name} — skilled in {', '.join(skills[:5]) if skills else '—'}. Recent role: {role or '—'}."

@st.cache_resource
def load_models():
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

# ---- UI ----
st.title("📑 AI Resume Parser")
st.caption("Semantic parsing with BERT NER, extractive summaries, JD fit, and a visual timeline.")

uploaded = st.file_uploader("Upload a resume", type=["pdf","docx","doc","png","jpg","jpeg","bmp","tiff"])
anonymize_flag = st.checkbox("Anonymize PII", value=False)

if uploaded and st.button("🚀 Parse Resume"):
    raw = uploaded.read()
    filename = uploaded.name
    text = extract_text_smart_bytes(raw, filename)

    if not text or len(text.strip()) < 10:
        st.error("Could not extract text. If scanned PDF, try uploading as image (PNG/JPG).")
    else:
        # models
        ner_model, sent_model = load_models()
        ner_results = run_ner(ner_model, text[:20000])

        # summary & skills
        summary = extractive_summary(sent_model, text)
        skills  = sorted({s for s in CURATED_SKILLS if s in text.lower()})

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

        # resume score (toy but intuitive)
        resume_score = min(100, 30 + len(skills)*5 + (10 if emails else 0) + (10 if phones else 0))

        # timeline
        timeline = detect_timeline(text)

                # JD fit (skill overlap + semantic similarity)
        jd_text_clean = (jd_text or "").strip()
        jd_skills = sorted({s for s in CURATED_SKILLS if s in jd_text_clean.lower()})
        missing_skills = [s for s in jd_skills if s not in skills]

        # semantic similarity
        sim = 0.0
        if jd_text_clean:
            cand_profile = " ".join([
                " ".join(skills), summary, text[:1200]
            ])
            emb_cand = sent_model.encode(cand_profile, convert_to_tensor=True)
            emb_jd   = sent_model.encode(jd_text_clean, convert_to_tensor=True)
            sim = float(util.cos_sim(emb_cand, emb_jd).item())

        # blend skill overlap + semantic sim
        skill_overlap = 0.0
        if jd_skills:
            skill_overlap = 1.0 - (len(missing_skills) / len(jd_skills))
        fit_score = round(100 * (jd_weight_skills * skill_overlap + jd_weight_text * sim))

        # suggestion
        if not jd_text_clean:
            suggestion = "ℹ️ Add a JD to get a fit suggestion."
        elif fit_score >= 75 and len(missing_skills) <= 2:
            suggestion = "✅ Strong fit — proceed to interview."
        elif fit_score >= 55:
            suggestion = "🟡 Borderline — consider technical screen."
        else:
            suggestion = "❌ Low fit — keep in pipeline."

        # ---- Hero Card ----
        st.markdown(f"""
        <div class="hero">
          <h2 style="margin-bottom:0;">{name_guess}</h2>
          <p class="small">📊 Resume Score: <b>{resume_score}/100</b></p>
          <p>💡 {elevator_pitch(name_guess, skills, '')}</p>
        </div>
        """, unsafe_allow_html=True)

        # JD fit box
        st.markdown(f"""
        <div style="margin-top:0.5rem; padding:1rem; background:#f6faff; border:1px solid #e6f0ff; border-radius:12px;">
          <b>JD Fit Score:</b> {fit_score}/100<br/>
          {suggestion}<br/>
          <span style="color:#666;">Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(fit_score)

        # ---- Metrics ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Skills", str(len(skills)))
        c2.metric("Emails", str(len(emails)))
        c3.metric("Links", str(len(links)))

        # ---- Tabs ----
        tabs = st.tabs(["Overview","Skills","Entities","Timeline","JSON"])

        with tabs[0]:
            st.subheader("Overview")
            st.write("**Emails:**", emails or "—")
            st.write("**Phones:**", phones or "—")
            st.write("**Links:**", links or "—")
            st.success("**Summary:** " + (summary or "—"))

        with tabs[1]:
            st.subheader("Skills")
            st.write("Detected Skills:", skills or "—")
            st.write("Grouped:", group_skills(skills))

        with tabs[2]:
            st.subheader("NER Entities (BERT)")
            if ner_results:
                st.dataframe(pd.DataFrame(ner_results))
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
                    chart = alt.Chart(df_t).mark_bar(size=18).encode(
                        x=alt.X('start_dt:T', title='Start'),
                        x2=alt.X2('end_dt:T', title='End'),
                        y=alt.Y('label:N', sort=None, title='Role / Company'),
                        tooltip=['role','company','start_raw','end_raw']
                    ).properties(height=max(200, 40 * len(df_t)))
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.write("Timeline table:", df_t)
            else:
                st.info("No timeline detected (need date ranges like 'Jun 2022 - Dec 2022').")

        with tabs[4]:
            st.subheader("JSON Export")
            parsed = {
                "filename": filename,
                "name_guess": name_guess,
                "emails": emails,
                "phones": phones,
                "links": links,
                "skills": skills,
                "skills_grouped": group_skills(skills),
                "summary": summary,
                "resume_score": resume_score,
                "ner_entities": ner_results,
                "timeline": timeline,
                "jd": {
                    "text_present": bool(jd_text_clean),
                    "jd_skills": jd_skills,
                    "missing_skills": missing_skills,
                    "semantic_similarity": round(sim, 3),
                    "fit_score": fit_score,
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
