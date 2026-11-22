import streamlit as st
from pypdf import PdfReader

# 1. Configure browser tab title, icon, and full-width layout
st.set_page_config(
    page_title="GlassBox AI — RAG Wizard (Behavioural Analysis)",
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
        "Select pre-loaded open-source behavioral psychology modules (adapted from OpenStax Psychology 2e, CC BY) "
        "or upload any open-source `.pdf` / `.txt` file on human behavior, personality traits, or nonverbal cues."
    )

    DEFAULT_DOCS = {
        "Big_Five_OCEAN_Traits.pdf (OpenStax Psych 2e)": (
            "[Page 1] The Five Factor Model (OCEAN) measures five core personality dimensions on a continuum: "
            "Openness to experience (imagination, curiosity, openness to new ideas), Conscientiousness (competence, "
            "self-discipline, thoughtfulness, goal-driven behavior), Extraversion (sociability, assertiveness, emotional "
            "expression), Agreeableness (cooperative, trustworthy, good-natured), and Neuroticism (tendency toward "
            "unstable emotions, anxiety, and stress reactivity). High conscientiousness strongly predicts workplace reliability."
        ),
        "Nonverbal_Kinesics_&_Pacifiers.pdf (Behavioral Guide)": (
            "[Page 1] Accurate body language analysis requires establishing a person's baseline behavior first and "
            "evaluating clusters of gestures rather than single isolated movements. Pacifying behaviors—such as neck "
            "touching, lip compression, self-hugging, or repetitive foot tapping—reflect autonomic nervous system arousal "
            "and psychological discomfort or stress, not definitive proof of deception. Ventral fronting (angling the "
            "torso and feet toward a speaker) signals genuine engagement and psychological comfort."
        ),
        "Social_Influence_&_Cognitive_Bias.pdf (OpenStax Psych 2e)": (
            "[Page 1] Fundamental Attribution Error occurs when observers overemphasize internal personality traits and "
            "underestimate situational factors when explaining another person's behavior. In contrast, the Actor-Observer "
            "Bias leads individuals to attribute their own mistakes to external circumstances while blaming others' "
            "mistakes on character flaws. Cognitive dissonance arises when actions conflict with core self-beliefs, "
            "prompting rationalization."
        ),
        "HEXACO_Honesty_Humility_Model.pdf (Trait Research)": (
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
        "Or upload an open-source PDF / TXT file (e.g., OpenStax Psychology 2e chapter):",
        type=["pdf", "txt"],
    )

    active_docs = {name: DEFAULT_DOCS[name] for name in selected_names}
    pdf_pages_loaded = 0

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
                # Preview first 1,200 chars cleanly if a huge PDF is uploaded
                preview = content if len(content) <= 1200 else content[:1200] + "\n\n... [truncated preview — full text passed to Step 2]"
                st.write(preview)

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=len(active_docs) > 0)

# =========================================================
# STEP 2 PLACEHOLDER (NEXT LESSON)
# =========================================================
elif step == 2:
    st.subheader("Step 2: Text Chunking & Overlap (Next Lesson)")
    loaded_docs = st.session_state.rag_data.get("docs", {})
    st.info(
        f"Ready for chunking! Step 1 passed **{len(loaded_docs)} source(s)** "
        f"({sum(len(t.split()) for t in loaded_docs.values()):,} words) into `st.session_state.rag_data['docs']`."
    )
    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=False)
