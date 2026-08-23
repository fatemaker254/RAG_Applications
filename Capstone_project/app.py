"""
Streamlit frontend for the QA Test Case Generation Agent.

Run with:
    streamlit run app.py
"""

import os
import json
import streamlit as st

from config import DATA_DIR, CHROMA_DIR
from pdf_ingest import load_and_chunk_all_pdfs
from vectorstore import build_vectorstore, load_vectorstore
from generate import generate_test_cases, NotARequirementError

st.set_page_config(
    page_title="QA Test Case Generation Agent",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }

    .app-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .app-subheader {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .tc-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .tc-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .tc-id {
        font-family: monospace;
        color: #6b7280;
        font-size: 0.85rem;
    }
    .tc-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin-top: 0.15rem;
    }
    .priority-badge {
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .priority-High { background: #fee2e2; color: #b91c1c; }
    .priority-Medium { background: #fef3c7; color: #b45309; }
    .priority-Low { background: #dcfce7; color: #15803d; }

    .tc-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #9ca3af;
        margin-top: 0.7rem;
        margin-bottom: 0.2rem;
    }
    .tc-body { color: #374151; font-size: 0.92rem; line-height: 1.5; }

    .context-chip {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "kb_status" not in st.session_state:
    # Try to load an existing knowledge base if one was already built
    try:
        st.session_state.collection = load_vectorstore()
        st.session_state.kb_status = f"Loaded existing knowledge base ({st.session_state.collection.count()} chunks)."
    except Exception:
        st.session_state.kb_status = "No knowledge base yet — build one from the sidebar."
if "test_cases" not in st.session_state:
    st.session_state.test_cases = None
if "context_chunks" not in st.session_state:
    st.session_state.context_chunks = None

# ---------------------------------------------------------------------------
# Sidebar: knowledge base management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("###  Knowledge Base")
    st.caption(st.session_state.kb_status)

    os.makedirs(DATA_DIR, exist_ok=True)
    existing_pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if existing_pdfs:
        st.markdown("**PDFs in `data/`:**")
        for f in existing_pdfs:
            st.markdown(f"- {f}")
    else:
        st.info("No PDFs in the data folder yet.")

    uploaded = st.file_uploader("Add requirement PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            with open(os.path.join(DATA_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to data/. Click Build below to ingest them.")
        st.rerun()

    st.markdown("---")
    if st.button(" Build / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Extracting and chunking PDFs..."):
            chunks = load_and_chunk_all_pdfs()
        if not chunks:
            st.error("No chunks extracted. Add PDFs to the data/ folder first.")
        else:
            progress = st.progress(0, text="Embedding chunks...")

            # Lightweight wrapper so the Streamlit progress bar updates
            # while build_vectorstore does its work chunk by chunk.
            import vectorstore as vs_module

            original_get_embedding = vs_module.get_embedding
            total = len(chunks)
            counter = {"i": 0}

            def tracked_get_embedding(text, *args, **kwargs):
                counter["i"] += 1
                progress.progress(min(counter["i"] / total, 1.0), text=f"Embedding chunk {counter['i']}/{total}")
                return original_get_embedding(text, *args, **kwargs)

            vs_module.get_embedding = tracked_get_embedding
            try:
                collection = build_vectorstore(chunks)
            finally:
                vs_module.get_embedding = original_get_embedding

            st.session_state.collection = collection
            st.session_state.kb_status = f"Knowledge base ready ({collection.count()} chunks)."
            progress.empty()
            st.success("Knowledge base built successfully.")
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header"> QA Test Case Generation Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subheader">RAG-powered test case generation from your requirement documents, '
    "running fully locally via Ollama.</div>",
    unsafe_allow_html=True,
)

requirement = st.text_area(
    "Requirement or user story to generate test cases for",
    placeholder='e.g. "Citizen registers a complaint through the Registration module"',
    height=100,
)

generate_clicked = st.button("Generate Test Cases", type="primary", disabled=(st.session_state.collection is None))

if st.session_state.collection is None:
    st.warning("Build the knowledge base from the sidebar before generating test cases.")

if generate_clicked and requirement.strip():
    with st.spinner("Retrieving context and generating test cases..."):
        try:
            test_cases, context_chunks = generate_test_cases(st.session_state.collection, requirement)
            st.session_state.test_cases = test_cases
            st.session_state.context_chunks = context_chunks
        except NotARequirementError as e:
            st.warning(str(e))
            st.session_state.test_cases = None
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.session_state.test_cases = None

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.test_cases:
    if st.session_state.context_chunks:
        st.markdown("**Context used:**")
        chips = "".join(
            f'<span class="context-chip">{c["source"]} · {c["section"]}</span>'
            for c in st.session_state.context_chunks
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.write("")

    for tc in st.session_state.test_cases:
        priority = tc.get("priority", "Medium")
        steps_html = "".join(f"<li>{step}</li>" for step in tc.get("steps", []))
        st.markdown(
            f"""
            <div class="tc-card">
                <div class="tc-header">
                    <div>
                        <div class="tc-id">{tc.get('id', '')}</div>
                        <div class="tc-title">{tc.get('title', '')}</div>
                    </div>
                    <span class="priority-badge priority-{priority}">{priority}</span>
                </div>
                <div class="tc-label">Preconditions</div>
                <div class="tc-body">{tc.get('preconditions', '')}</div>
                <div class="tc-label">Steps</div>
                <ol class="tc-body">{steps_html}</ol>
                <div class="tc-label">Expected Result</div>
                <div class="tc-body">{tc.get('expected_result', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.download_button(
        "⬇ Download as JSON",
        data=json.dumps(st.session_state.test_cases, indent=2),
        file_name="test_cases.json",
        mime="application/json",
    )