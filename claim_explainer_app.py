# ============================================================
# 💎 Insurance Claim Explanation App - Using DocTR (No EasyOCR/Tesseract)
# ============================================================

import streamlit as st
import pdfplumber
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import pipeline
import tempfile
import os
from pathlib import Path

# ============================================================
# 🎨 Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="Insurance Claim Explainer",
    page_icon="🧾",
    layout="centered",
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(180deg, #eef2f3, #ffffff);
    }
    .main-title {
        text-align: center;
        font-size: 36px !important;
        color: #1f4e79;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .result-box {
    background-color: #f9fafb;
    color: #1f2937;  /* 🟢 Dark gray text color for readability */
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    white-space: pre-wrap;  /* Keep text formatting */
    font-size: 16px;
    line-height: 1.5;

    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🧠 Load Models
# ============================================================

@st.cache_resource
def load_models():
    ocr_model = ocr_predictor(pretrained=True)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return ocr_model, summarizer

ocr_model, summarizer = load_models()

# ============================================================
# 🧩 Helper Functions
# ============================================================

def extract_text_from_pdf(pdf_path):
    """Extract text using pdfplumber or OCR fallback"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    if text.strip():
        return text.strip()
    
    # Fallback to OCR if PDF text not selectable
    doc = DocumentFile.from_pdf(pdf_path)
    result = ocr_model(doc)
    return result.render()

def extract_text_from_image(image_path):
    """Extract text from image using DocTR OCR"""
    doc = DocumentFile.from_images(image_path)
    result = ocr_model(doc)
    return result.render()

def summarize_claim(text):
    """Summarize insurance claim text"""
    if not text.strip():
        return "⚠️ No text detected in the document."
    text = text[:3000]  # avoid model limit overflow
    result = summarizer(text, max_length=130, min_length=40, do_sample=False)
    return result[0]['summary_text'].strip()

def process_claim_file(file_path):
    """Main pipeline: Extract → Summarize"""
    file_ext = Path(file_path).suffix.lower()
    if file_ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif file_ext in [".jpg", ".jpeg", ".png"]:
        raw_text = extract_text_from_image(file_path)
    elif file_ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        return "⚠️ Unsupported file type."

    if not raw_text.strip():
        return "⚠️ Could not extract any readable text."
    return summarize_claim(raw_text)

# ============================================================
# 🖥️ Streamlit UI
# ============================================================

st.markdown('<h1 class="main-title">🧾 Insurance Claim Explanation</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Summarize claim PDFs or images into simple English.</p>', unsafe_allow_html=True)

option = st.sidebar.radio("Select Input Mode:", ["📝 Enter Text", "📂 Upload File"])

if option == "📝 Enter Text":
    claim_text = st.text_area("✍️ Enter or paste claim details below:", height=180)
    if st.button("🔍 Explain Claim"):
        if claim_text.strip():
            with st.spinner("Summarizing..."):
                summary = summarize_claim(claim_text)
            st.success("✅ Claim summarized successfully!")
            st.markdown(f'<div class="result-box"><h4>🧾 Customer-Friendly Explanation</h4><p>{summary}</p></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text.")
else:
    uploaded_file = st.file_uploader("📎 Upload a file (.pdf, .jpg, .jpeg, .png, .txt):", type=["pdf", "jpg", "jpeg", "png", "txt"])
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info(f"✅ File uploaded: **{uploaded_file.name}**")

        if st.button("🔍 Explain Claim"):
            with st.spinner("Extracting and summarizing..."):
                summary = process_claim_file(temp_path)
            st.success("✅ Claim summarized successfully!")
            st.markdown(f'<div class="result-box"><h4>🧾 Customer-Friendly Explanation</h4><p>{summary}</p></div>', unsafe_allow_html=True)

        os.remove(temp_path)
