"""
Streamlit UI — Invoice Parser using Groq LLaMA Vision Model.

Run with:
    streamlit run d:/invoice_parser/vision_model/streamlit_app.py
"""

import json
import streamlit as st
from io import BytesIO
from PIL import Image

from vision_core import process_document, pdf_bytes_to_images

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Invoice Parser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Dark premium theme overrides ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.hero p {
    color: #a0aec0;
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
}

/* Result cards */
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.result-card h3 {
    color: #7c3aed;
    margin-top: 0;
}

/* Status badges */
.badge-success {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    color: #10b981;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
}
.badge-error {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Sidebar styles */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #a0aec0;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(124,58,237,0.4);
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Vision Invoice Parser</h1>
    <p>Upload a PDF or image invoice — powered by Groq LLaMA Vision for OCR and LLM for structured extraction.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. **Upload** a PDF or image invoice
    2. **Vision model** (`LLaMA 4 Scout`) reads all text
    3. **LLM** (`LLaMA 3.1 8B`) extracts structured JSON
    4. **Review** the raw text and JSON output
    """)
    st.divider()
    st.markdown("### 🧠 Models Used")
    st.code("Vision: meta-llama/llama-4-scout-17b-16e-instruct\nJSON  : llama-3.1-8b-instant", language="text")
    st.divider()
    st.caption("Powered by Groq API • Built with Streamlit")

# ── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📎 Upload an invoice (PDF or Image)",
    type=["pdf", "jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    help="Supported: PDF, JPG, JPEG, PNG, WEBP, BMP, TIFF",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    # ── Preview ──────────────────────────────────────────────────────────────
    col_preview, col_action = st.columns([1, 1])

    with col_preview:
        st.markdown("#### 📄 Preview")
        if filename.lower().endswith(".pdf"):
            try:
                pages = pdf_bytes_to_images(file_bytes, dpi=150)
                if pages:
                    st.image(pages[0], caption=f"Page 1 of {len(pages)}", use_column_width=True)
            except Exception:
                st.info("PDF preview unavailable.")
        else:
            st.image(file_bytes, caption=filename, use_column_width=True)

    with col_action:
        st.markdown("#### 📋 File Details")
        st.markdown(f"- **Name:** `{filename}`")
        st.markdown(f"- **Size:** `{len(file_bytes) / 1024:.1f} KB`")
        st.markdown(f"- **Type:** `{uploaded_file.type}`")

        st.markdown("---")

        parse_btn = st.button("🚀 Extract & Parse", type="primary")

    # ── Processing ───────────────────────────────────────────────────────────
    if parse_btn:
        with st.spinner("🔍 Extracting text with Vision model..."):
            try:
                raw_text, structured_data = process_document(file_bytes, filename)

                # ── Raw text output ──────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 📝 Extracted Raw Text")
                st.markdown('<span class="badge-success">✓ Extraction Complete</span>', unsafe_allow_html=True)
                st.text_area(
                    "Raw OCR Output",
                    value=raw_text,
                    height=300,
                    label_visibility="collapsed",
                )
                # ── JSON output ──────────────────────────────────────────────
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Structured Invoice JSON")

                if structured_data:
                    st.markdown('<span class="badge-success">✓ Parsing Complete</span>', unsafe_allow_html=True)
                    st.json(structured_data, expanded=True)

                    # Download button
                    json_str = json.dumps(structured_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_str,
                        file_name=f"{filename.rsplit('.', 1)[0]}_parsed.json",
                        mime="application/json",
                    )
                else:
                    st.markdown('<span class="badge-error">✗ JSON parsing failed</span>', unsafe_allow_html=True)
                    st.warning("The LLM did not return valid JSON. The raw text above may still be useful.")

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Check that your `GROQ_API_KEY` is set correctly in the `.env` file.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Vision Invoice Parser • Groq LLaMA •")
