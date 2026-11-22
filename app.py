import re
import pandas as pd
import streamlit as st
from pypdf import PdfReader

# 1. Configure browser tab title, icon, and full-width layout
st.set_page_config(
    page_title="GlassBox AI — Behavioral RAG Wizard",
    page_icon="🧠",
    layout="wide",
)

# 2. Initialize Wizard Memory (Session State)
if "rag_step" not in st.session_state:
    st.session_state.rag_step = 1

if "rag_data" not in st.session_state:
    st.session_state.rag_data = {}


# 3. Navigation Helper Functions
def reset_wizard():
    st.session_state.rag_step = 1
    st.session_state.rag_data = {}


def render_wizard_header(steps: list[str]):
    st.title("🧠 RAG Pipeline — Glass-Box Inspection Wizard")
    st.caption("Project: Human Behavior, Personality Traits & Body Language Advisor")

    current = st.session_state.rag_step
    total = len(steps)

    st.progress(current / total, text=f"Step {current} of {total}: **{steps[current - 1]}**")

    cols = st.columns(total)
    for idx, label in enumerate(steps, start=1):
        with cols[idx - 1]:
            if idx < current:
                st.success(f"✓ {idx}. {label}")
            elif idx == current:
                st.info(f"▶ {idx}. {label}")
            else:
                st.markdown(f"<span style='color:gray'>○ {idx}. {label}</span>", unsafe_allow_html=True)
    st.divider()


def render_wizard_nav(total_steps: int, can_proceed: bool = True):
    st.divider()
    col_cancel, col_spacer, col_back, col_next = st.columns([1.4, 4.6, 1.2, 1.4])

    with col_cancel:
        if st.button("✖ Cancel", use_container_width=True):
            reset_wizard()
            st.rerun()

    with col_back:
        if st.session_state.rag_step > 1:
            if st.button("◀ Back", use_container_width=True):
                st.session_state.rag_step -= 1
                st.rerun()

    with col_next:
        if st.session_state.rag_step < total_steps:
            if st.button("Next ▶", type="primary", disabled=not can_proceed, use_container_width=True):
                st.session_state.rag_step += 1
                st.rerun()
        else:
            if st.button("✔ Finish", type="primary", use_container_width=True):
                reset_wizard()
                st.rerun()


def extract_pdf_text(uploaded_pdf) -> tuple[str, int]:
    """Reads an uploaded PDF stream and returns (formatted_text, page_count)."""
    reader = PdfReader(uploaded_pdf)
    pages_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_page = page.extract_text() or ""
        cleaned = " ".join(raw_page.split())
        if cleaned:
            pages_text.append(f"[Page {page_num}] {cleaned}")
    return "\n\n".join(pages_text), len(reader.pages)


# 4. Define the 5 Stages of Our RAG Pipeline
RAG_STEPS = [
    "1. Ingest PDF / Docs",
    "2. Chunking & Overlap",
    "3. Vector Embeddings",
    "4. Top-K Retrieval",
    "5. Grounded Answer",
]

render_wizard_header(RAG_STEPS)
step = st.session_state.rag_step

# =========================================================
# STEP 1: DOCUMENT & PDF INGESTION
# =========================================================
if step == 1:
    st.subheader("Step 1: Load Behavioral Psychology Docs or Upload a PDF")
    st.write(
        "Select pre-loaded behavioral psychology reference modules or upload any open-access "
        "`.pdf` / `.txt` file on human behavior, personality traits, or nonverbal cues."
    )

    DEFAULT_DOCS = {
        "Big_Five_OCEAN_Traits.txt": (
            "[Page 1] The Five Factor Model (OCEAN) measures five core personality dimensions on a continuum: "
            "Openness to experience (imagination, curiosity, openness to new ideas), Conscientiousness (competence, "
            "self-discipline, thoughtfulness, goal-driven behavior), Extraversion (sociability, assertiveness, emotional "
            "expression), Agreeableness (cooperative, trustworthy, good-natured), and Neuroticism (tendency toward "
            "unstable emotions, anxiety, and stress reactivity). High conscientiousness strongly predicts workplace reliability."
        ),
        "Nonverbal_Kinesics_Pacifiers.txt": (
            "[Page 1] Accurate body language analysis requires establishing a person's baseline behavior first and "
            "evaluating clusters of gestures rather than single isolated movements. Pacifying behaviors—such as neck "
            "touching, lip compression, self-hugging, or repetitive foot tapping—reflect autonomic nervous system arousal "
            "and psychological discomfort or stress, not definitive proof of deception. Ventral fronting (angling the "
            "torso and feet toward a speaker) signals genuine engagement and psychological comfort."
        ),
        "Social_Influence_Cognitive_Bias.txt": (
            "[Page 1] Fundamental Attribution Error occurs when observers overemphasize internal personality traits and "
            "underestimate situational factors when explaining another person's behavior. In contrast, the Actor-Observer "
            "Bias leads individuals to attribute their own mistakes to external circumstances while blaming others' "
            "mistakes on character flaws. Cognitive dissonance arises when actions conflict with core self-beliefs, "
            "prompting rationalization."
        ),
        "HEXACO_Honesty_Humility.txt": (
            "[Page 1] The HEXACO model expands the Big Five by adding a sixth factor: Honesty-Humility, encompassing "
            "sincerity, fairness, greed avoidance, and modesty. Individuals scoring low on Honesty-Humility are more "
            "likely to manipulate others for personal gain, feel entitled to special treatment, and bend ethical rules "
            "when oversight is weak."
        ),
    }

    selected_names = st.multiselect(
        "Choose built-in behavioral & trait modules:",
        options=list(DEFAULT_DOCS.keys()),
        default=list(DEFAULT_DOCS.keys())[:3],
    )

    uploaded_file = st.file_uploader(
        "Or upload an open-source PDF / TXT file:",
        type=["pdf", "txt"],
    )

    active_docs = {name: DEFAULT_DOCS[name] for name in selected_names}

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".pdf"):
            extracted_text, pdf_pages_loaded = extract_pdf_text(uploaded_file)
            if extracted_text:
                active_docs[f"{uploaded_file.name} ({pdf_pages_loaded} pages)"] = extracted_text
            else:
                st.error("No extractable text found in this PDF (it may be a scanned image without OCR).")
        else:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            active_docs[uploaded_file.name] = raw_text

    st.session_state.rag_data["docs"] = active_docs

    # Glass-Box Output Inspection
    st.markdown("#### 🔍 Step 1 Output Inspection (Extracted Corpus)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Sources", len(active_docs))
    c2.metric("Total Characters", f"{sum(len(t) for t in active_docs.values()):,}")
    c3.metric("Total Words", f"{sum(len(t.split()) for t in active_docs.values()):,}")

    if not active_docs:
        st.warning("Select at least one behavioral module or upload a PDF to enable Next ▶.")
    else:
        for doc_name, content in active_docs.items():
            with st.expander(f"📄 {doc_name} — {len(content):,} characters", expanded=True):
                preview = (
                    content
                    if len(content) <= 1200
                    else content[:1200] + "\n\n... [truncated preview — full text passed to Step 2]"
                )
                st.write(preview)

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=len(active_docs) > 0)

# =========================================================
# STEP 2: SLIDING-WINDOW TEXT CHUNKING & OVERLAP
# =========================================================
elif step == 2:
    st.subheader("Step 2: Sliding-Window Text Chunking & Overlap")
    st.write(
        "Adjust **Chunk Size** and **Chunk Overlap** below to watch how your documents split into "
        "searchable passages while preserving context across chunk boundaries."
    )

    docs = st.session_state.rag_data.get("docs", {})

    # Interactive Hyperparameter Controls
    col1, col2 = st.columns(2)
    chunk_size = col1.slider(
        "Chunk Size (characters per chunk):",
        min_value=120,
        max_value=500,
        value=240,
        step=20,
    )
    overlap = col2.slider(
        "Chunk Overlap (shared characters between neighbors):",
        min_value=0,
        max_value=120,
        value=50,
        step=10,
    )

    # Guardrail: overlap must always be smaller than chunk_size
    effective_overlap = min(overlap, chunk_size - 20)
    stride = chunk_size - effective_overlap

    chunks = []
    for doc_name, text in docs.items():
        start = 0
        chunk_num = 1
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                # Track the most recent [Page N] marker seen up to this slice
                page_matches = re.findall(r"\[Page (\d+)\]", text[: start + chunk_size])
                page_label = f"Page {page_matches[-1]}" if page_matches else "Page 1"

                chunks.append(
                    {
                        "chunk_id": f"{doc_name.split('.')[0]}#C{chunk_num}",
                        "source": doc_name,
                        "page": page_label,
                        "chars": len(piece),
                        "text": piece,
                    }
                )
                chunk_num += 1

            # Stop once the window reaches the end of the document
            if start + chunk_size >= len(text):
                break
            start += stride

    # Save generated chunks into session_state for Step 3
    st.session_state.rag_data["chunks"] = chunks

    # Glass-Box Output Inspection
    st.markdown("#### 🔍 Step 2 Output Inspection (Chunked Corpus)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Chunks Generated", len(chunks))
    avg_chars = round(sum(c["chars"] for c in chunks) / len(chunks)) if chunks else 0
    m2.metric("Avg. Chunk Length", f"{avg_chars} chars")
    m3.metric("Window Stride", f"{stride} chars / step")

    # Visual Proof of Overlap between Chunk 1 and Chunk 2 of the first document
    if len(chunks) >= 2 and chunks[0]["source"] == chunks[1]["source"] and effective_overlap > 0:
        shared_tail = chunks[0]["text"][-effective_overlap:]
        with st.expander("🔗 Visual Proof: How Overlap Bridges Chunk #1 and Chunk #2", expanded=True):
            st.caption(
                f"Notice how the last ~{effective_overlap} characters of **{chunks[0]['chunk_id']}** repeat at the "
                f"beginning of **{chunks[1]['chunk_id']}** so sentences aren't severed:"
            )
            c_left, c_right = st.columns(2)
            c_left.info(f"**{chunks[0]['chunk_id']} (End):**\n\n...{chunks[0]['text'][-110:]}")
            c_right.success(f"**{chunks[1]['chunk_id']} (Start):**\n\n{chunks[1]['text'][:110]}...")
            st.code(f"Shared Overlap Bridge: \"{shared_tail}\"", language="text")

    # Full Searchable Chunk Table
    st.dataframe(
        pd.DataFrame(chunks),
        use_container_width=True,
        hide_index=True,
    )

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=len(chunks) > 0)

# =========================================================
# STEP 3 PLACEHOLDER (NEXT LESSON)
# =========================================================
elif step == 3:
    st.subheader("Step 3: Vector Embeddings & Semantic Space (Next Lesson)")
    chunk_count = len(st.session_state.rag_data.get("chunks", []))
    st.info(f"Success! Step 2 passed **{chunk_count} chunks** into `st.session_state.rag_data['chunks']`.")
    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=False)
