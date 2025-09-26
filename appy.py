# appy (clean, Streamlit Cloud-friendly)
import streamlit as st
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text
import docx
from PIL import Image
import pytesseract
import re
import nltk

# make sure both punkt and punkt_tab are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import dateparser
import pandas as pd
import altair as alt
import json
from datetime import datetime
from dateutil import parser as dateparser2

nltk.download('punkt', quiet=True)

st.set_page_config(layout="wide", page_title="AI Resume Parser")

# ---------- Helpers ----------
def extract_text_from_pdf_bytes(file_bytes):
    """
    Try pdfminer-based extraction. If the PDF is a scanned image inside PDF,
    this function may return empty; in that case please upload an image (png/jpg)
    or run a conversion externally (pdf -> images).
    """
    with BytesIO(file_bytes) as f:
        try:
            text = pdf_extract_text(f)
        except Exception:
            text = ""
    return text

def extract_text_from_docx_bytes(file_bytes):
    doc = docx.Document(BytesIO(file_bytes))
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)

def extract_text_from_image_bytes(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text

def extract_text_smart_bytes(file_bytes, filename):
    fname = filename.lower()
    if fname.endswith(".pdf"):
        txt = extract_text_from_pdf_bytes(file_bytes)
        # If pdfminer fails (empty or short), return empty so user can try uploading image
        return txt or ""
    elif fname.endswith((".docx",".doc")):
        return extract_text_from_docx_bytes(file_bytes)
    elif fname.endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
        return extract_text_from_image_bytes(file_bytes)
    else:
        # try decoding as text
        try:
            return file_bytes.decode("utf-8")
        except:
            return file_bytes.decode("latin-1", errors="ignore")

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s-]?)?(\d{10}|\d{3}[\s-]\d{3}[\s-]\d{4}|\(\d{3}\)\s*\d{3}-\d{4})')
LINK_RE = re.compile(r'(https?://\S+|linkedin\.com/\S+|github\.com/\S+)')

CURATED_SKILLS = [
    "python","java","c++","c","javascript","react","angular","node","django","flask",
    "tensorflow","pytorch","keras","sql","postgresql","mysql","mongodb","docker",
    "kubernetes","aws","azure","gcp","pandas","numpy","scikit-learn","excel","git","communication"
]

@st.cache_resource
def load_models():
    # Pretrained BERT-based NER (token-classifier) and a sentence-transformer for embeddings
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    sent_model = SentenceTransformer('all-MiniLM-L6-v2')
    return ner, sent_model

def run_ner(ner_model, text):
    if not text or len(text.strip()) < 10:
        return []
    try:
        ents = ner_model(text)
    except Exception:
        ents = []
    normalized = []
    for e in ents:
        label = e.get("entity_group") or e.get("entity") or e.get("label")
        normalized.append({
            "entity": label,
            "word": e.get("word") or e.get("entity"),
            "score": float(e.get("score", 0))
        })
    return normalized

def extractive_summary(sent_model, text, top_k=3):
    sentences = nltk.tokenize.sent_tokenize(text)
    if not sentences:
        return ""
    embeddings = sent_model.encode(sentences, convert_to_tensor=True)
    doc_emb = embeddings.mean(dim=0)
    cos_scores = util.cos_sim(doc_emb, embeddings)[0]
    top_results = cos_scores.argsort(descending=True)[:top_k].tolist()
    top_sentences = [sentences[idx] for idx in sorted(top_results)]
    return " ".join(top_sentences)

def detect_gaps(ranges, threshold_months=6):
    parsed = []
    for r in ranges:
        try:
            s = dateparser2.parse(r['start'])
            e = dateparser2.parse(r['end']) if r['end'] and r['end'].lower()!="present" else datetime.now()
            if s and e:
                parsed.append((s,e,r))
        except:
            pass
    parsed = sorted(parsed, key=lambda x: x[0])
    gaps = []
    for i in range(len(parsed)-1):
        diff = (parsed[i+1][0].year - parsed[i][1].year)*12 + (parsed[i+1][0].month - parsed[i][1].month)
        if diff >= threshold_months:
            gaps.append({"gap_months": diff, "between": (parsed[i][2], parsed[i+1][2])})
    return gaps

def group_skills(skills):
    tech = []
    nontech = []
    for s in skills:
        if s in {"python","sql","tensorflow","docker","aws","git","react","django","pandas","numpy"}:
            tech.append(s)
        else:
            nontech.append(s)
    return {"technical": tech, "non_technical": nontech}

def anonymize(parsed):
    out = dict(parsed)
    out['name_guess'] = "CANDIDATE_XXXX"
    out['emails'] = ["hidden@example.com"]*len(out.get('emails',[]))
    out['phones'] = ["hidden"]*len(out.get('phones',[]))
    return out

def elevator_pitch(parsed):
    top_skills = parsed.get("skills", [])[:5]
    name = parsed.get("name_guess","Candidate")
    role = "recent role"
    if parsed.get("timeline"):
        try:
            role = parsed['timeline'][-1].get("role") or role
        except:
            pass
    return f"{name} — skilled in {', '.join(top_skills)}. Recent role: {role}."

def sanitize_for_json(o):
    # convert some objects to JSON-safe types
    if isinstance(o, dict):
        return {k: sanitize_for_json(v) for k,v in o.items()}
    if isinstance(o, list):
        return [sanitize_for_json(x) for x in o]
    if isinstance(o, datetime):
        return o.isoformat()
    return o

# ---------- Streamlit UI ----------
st.title("📑 AI Resume Parser (Streamlit Cloud ready)")
st.write("Upload resumes (PDF/DOCX/Image). Uses pretrained BERT NER + sentence-transformers for a demo-quality parser.")

uploaded = st.file_uploader("Upload resume", type=["pdf","docx","doc","png","jpg","jpeg","bmp","tiff"])
anonymize_flag = st.checkbox("Anonymize PII", value=False)

if uploaded and st.button("Parse Resume"):
    raw = uploaded.read()
    filename = uploaded.name
    text = extract_text_smart_bytes(raw, filename)
    if not text or len(text.strip()) < 10:
        st.error("Could not extract readable text. If this is a scanned PDF, please upload as PNG/JPG or convert PDF pages to images before uploading.")
    else:
        st.success("Text extracted. Running models...")
        with st.expander("📄 Raw Text"):
            st.text_area("Extracted text", text, height=250)

        ner_model, sent_model = load_models()
        ner_results = run_ner(ner_model, text[:30000])
        summary = extractive_summary(sent_model, text, top_k=3)
        skills = sorted({s for s in CURATED_SKILLS if s in text.lower()})

        emails = EMAIL_RE.findall(text)
        phones = []
        for m in PHONE_RE.findall(text):
            if isinstance(m, tuple):
                phones.append("".join(m).strip())
            else:
                phones.append(m)
        links = LINK_RE.findall(text)

        DATE_RANGE_RE = re.compile(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|\d{1,2})[^\n,-]{0,12}\s?\d{2,4})\s*(?:to|-|–|—)\s*((?:Present|present|[A-Za-z]{3,9}\s?\d{2,4}|\d{4}))', re.IGNORECASE)
        ranges = []
        for m in DATE_RANGE_RE.finditer(text):
            s_raw, e_raw = m.groups()
            s = dateparser.parse(s_raw)
            e = dateparser.parse(e_raw) if e_raw and e_raw.lower()!="present" else datetime.now()
            if s:
                ranges.append({"start_raw":s_raw,"end_raw":e_raw,"start":s.isoformat(),"end":e.isoformat(),"role":""})

        name_guess = ""
        for e in ner_results:
            if str(e.get('entity','')).lower() in ('per','person','person_name','per'):
                name_guess = e.get('word',''); break
        if not name_guess and emails:
            name_guess = emails[0].split("@")[0]

        parsed = {
            "filename": filename,
            "name_guess": name_guess,
            "emails": emails,
            "phones": phones,
            "links": links,
            "skills": skills,
            "skills_grouped": group_skills(skills),
            "summary": summary,
            "timeline": ranges,
            "ner_entities": ner_results
        }

        if anonymize_flag:
            parsed = anonymize(parsed)

        # UI tabs
        tabs = st.tabs(["Overview","Entities","Timeline","JSON"])
        with tabs[0]:
            st.subheader("Candidate Overview")
            st.write("**Name:**", parsed['name_guess'])
            st.write("**Emails:**", parsed['emails'] or "—")
            st.write("**Phones:**", parsed['phones'] or "—")
            st.write("**Links:**", parsed['links'] or "—")
            st.write("**Skills (grouped):**", parsed['skills_grouped'])
            st.info("**Elevator Pitch:** " + elevator_pitch(parsed))
            st.success(parsed['summary'] or "—")

        with tabs[1]:
            st.subheader("BERT NER Entities")
            if parsed['ner_entities']:
                df = pd.DataFrame(parsed['ner_entities'])
                st.dataframe(df)
            else:
                st.write("No entities detected.")

        with tabs[2]:
            st.subheader("Career Timeline")
            if parsed['timeline']:
                df_t = pd.DataFrame(parsed['timeline'])
                try:
                    df_t['start_dt'] = pd.to_datetime(df_t['start'])
                    df_t['end_dt'] = pd.to_datetime(df_t['end'])
                    chart = alt.Chart(df_t).mark_bar().encode(
                        x='start_dt:T', x2='end_dt:T',
                        y=alt.Y('start_raw:N', sort=None),
                        tooltip=['start_raw','end_raw']
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.write(df_t)
                gaps = detect_gaps(parsed['timeline'])
                if gaps:
                    st.warning(f"Detected career gaps: {gaps}")
            else:
                st.write("No timeline found.")

        with tabs[3]:
            st.subheader("JSON Export")
            st.json(sanitize_for_json(parsed))
            st.download_button("Download JSON", data=json.dumps(sanitize_for_json(parsed), indent=2), file_name="parsed_resume.json", mime="application/json")

st.caption("Demo: pretrained BERT NER + sentence-transformers. For scanned PDFs, upload images (PNG/JPG) or convert PDF pages to images first.")
