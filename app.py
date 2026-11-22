import math
import re
import time
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


# 3. Navigation & Helper Functions
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
            if st.button("✔ Finish & Restart", type="primary", use_container_width=True):
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


# 4. Semantic Embedding Engine (12-Dimensional Unit Vectors)
CONCEPT_AXES = [
    ("OCEAN_Traits", ["ocean", "five", "personality", "openness", "extraversion", "agreeableness", "trait", "traits"]),
    ("Conscientiousness", ["conscientiousness", "discipline", "reliability", "workplace", "goal", "competence"]),
    ("Neuroticism_Stress", ["neuroticism", "anxiety", "stress", "arousal", "unstable", "emotions", "discomfort"]),
    ("Pacifiers_Kinesics", ["pacifying", "pacifiers", "neck", "touching", "lip", "compression", "foot", "tapping", "hugging", "deception", "lie"]),
    ("Body_Posture", ["body", "language", "baseline", "clusters", "gestures", "ventral", "fronting", "torso", "feet", "engagement"]),
    ("Attribution_Bias", ["attribution", "fundamental", "error", "actor", "observer", "bias", "situational", "mistakes", "circumstances"]),
    ("Cognitive_Dissonance", ["cognitive", "dissonance", "conflict", "beliefs", "rationalization", "social", "influence"]),
    ("HEXACO_Ethics", ["hexaco", "honesty", "humility", "sincerity", "fairness", "greed", "modesty", "manipulate", "ethical", "entitled"]),
    ("General_Hash_1", []),
    ("General_Hash_2", []),
    ("General_Hash_3", []),
    ("General_Hash_4", []),
]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "into", "than", "their", "them", "they", "have", "has", "had",
}


def compute_embedding(text: str) -> list[float]:
    """Maps text into a 12-dimensional L2-normalized semantic vector."""
    words = [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]
    raw_vec = [0.0] * len(CONCEPT_AXES)

    for idx in range(8):
        axis_keywords = CONCEPT_AXES[idx][1]
        for w in words:
            if w in axis_keywords:
                raw_vec[idx] += 1.5

    for w in words:
        bucket = 8 + (sum(ord(ch) for ch in w) % 4)
        raw_vec[bucket] += 0.35

    norm = math.sqrt(sum(v * v for v in raw_vec)) or 1.0
    return [round(v / norm, 4) for v in raw_vec]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return round(max(0.0, min(1.0, dot)), 4)


def synthesize_grounded_answer(
    api_key: str,
    query: str,
    retrieved_chunks: list[dict],
) -> tuple[str, dict]:
    """Generates a strictly grounded answer from retrieved chunks."""
    start_time = time.perf_counter()

    if not retrieved_chunks:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        return (
            "⚠️ **Insufficient Evidence in Knowledge Base:** None of the indexed chunks scored above your "
            "similarity threshold. To prevent hallucination, the pipeline refused to guess. Try lowering the "
            "similarity threshold in Step 4 or uploading a relevant document in Step 1.",
            {"mode": "Guardrail Refusal", "latency_ms": elapsed_ms, "tokens_used": 0},
        )

    context_lines = [
        f"[{c['source']} | {c['page']} | Chunk {c['chunk_id']}]: {c['text']}"
        for c in retrieved_chunks
    ]
    context_block = "\n\n".join(context_lines)

    system_prompt = (
        "You are a behavioral psychology research assistant. Answer the user's question strictly using ONLY "
        "the provided context passages. Cite exact source filenames and page numbers in brackets after each claim. "
        "If the context does not contain the answer, state that clearly."
    )
    user_prompt = f"CONTEXT PASSAGES:\n{context_block}\n\nUSER QUESTION: {query}"

    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            total_tokens = resp.usage.total_tokens if resp.usage else 0
            return resp.choices[0].message.content, {
                "mode": "Live OpenAI (gpt-4o-mini)",
                "latency_ms": elapsed_ms,
                "tokens_used": total_tokens,
            }
        except Exception as exc:
            api_error_note = f"> *Note: OpenAI API call fell back to Zero-Key Synthesis ({exc})*\n\n"
    else:
        api_error_note = ""

    query_terms = {w for w in re.findall(r"[a-z]{3,}", query.lower()) if w not in STOPWORDS}
    cited_points = []
    unique_sources = []

    for chunk in retrieved_chunks:
        clean_passage = re.sub(r"\[Page \d+\]\s*", "", chunk["text"]).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_passage) if len(s.strip()) > 20]
        if not sentences:
            sentences = [clean_passage]

        best_sent = max(
            sentences,
            key=lambda s: len(query_terms & set(re.findall(r"[a-z]{3,}", s.lower()))),
        )
        # Capitalize cleanly if a chunk began mid-sentence after an overlap boundary
        if best_sent and best_sent[0].islower():
            best_sent = best_sent[0].upper() + best_sent[1:]

        citation = f"`{chunk['source']} ({chunk['page']})`"
        cited_points.append(f"- {best_sent} — *Source: {citation}*")
        if citation not in unique_sources:
            unique_sources.append(citation)

    answer_md = (
        f"{api_error_note}"
        f"Based strictly on the top **{len(retrieved_chunks)}** retrieved behavioral passages:\n\n"
        + "\n".join(cited_points)
        + f"\n\n**Verified Citations:** {', '.join(unique_sources)}"
    )
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    est_tokens = len((system_prompt + user_prompt + answer_md).split())
    return answer_md, {
        "mode": "Zero-Key Deterministic Grounded Mode",
        "latency_ms": elapsed_ms,
        "tokens_used": est_tokens,
    }


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Glass-Box Controls")
    api_key = st.text_input(
        "OpenAI API Key (Optional):",
        type="password",
        help="Leave blank to run the entire 5-step pipeline in Zero-Key Demo Mode.",
    )
    if api_key:
        st.success("Live OpenAI Mode Active")
    else:
        st.info("Zero-Key Demo Mode Active")

    st.markdown("---")
    
    if st.button("🔄 Reset Entire Wizard", use_container_width=True):
        reset_wizard()
        st.rerun()


# 5. Define the 5 Stages of Our RAG Pipeline
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
        default=list(DEFAULT_DOCS.keys()),
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

    st.markdown("#### 🔍 Step 1 Output Inspection (Extracted Corpus)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Sources", len(active_docs))
    c2.metric("Total Characters", f"{sum(len(t) for t in active_docs.values()):,}")
    c3.metric("Total Words", f"{sum(len(t.split()) for t in active_docs.values()):,}")

    if not active_docs:
        st.warning("Select at least one behavioral module or upload a PDF to enable Next ▶.")
    else:
        for doc_name, content in active_docs.items():
            with st.expander(f"📄 {doc_name} — {len(content):,} characters", expanded=False):
                preview = (
                    content
                    if len(content) <= 1200
                    else content[:1200] + "\n\n... [truncated preview — full text passed to Step 2]"
                )
                st.write(preview)

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=len(active_docs) > 0)

# =========================================================
# STEP 2: WORD-BOUNDARY SLIDING-WINDOW CHUNKING & OVERLAP
# =========================================================
elif step == 2:
    st.subheader("Step 2: Word-Boundary Text Chunking & Overlap")
    st.write(
        "Adjust **Chunk Size** and **Chunk Overlap** below. Notice how boundary snapping aligns cuts to "
        "whole words so words are never sliced in half across chunk edges."
    )

    docs = st.session_state.rag_data.get("docs", {})

    col1, col2 = st.columns(2)
    chunk_size = col1.slider("Chunk Size (characters per chunk):", 120, 500, 240, 20)
    overlap = col2.slider("Chunk Overlap (shared characters between neighbors):", 0, 120, 50, 10)

    effective_overlap = min(overlap, chunk_size - 20)

    chunks = []
    for doc_name, text in docs.items():
        start = 0
        chunk_num = 1
        while start < len(text):
            end = min(len(text), start + chunk_size)

            # 1. Snap end backward to nearest space so we never cut a word in half
            if end < len(text):
                last_space = text.rfind(" ", start + 40, end)
                if last_space != -1:
                    end = last_space

            piece = text[start:end].strip()
            if piece:
                page_matches = re.findall(r"\[Page (\d+)\]", text[:end])
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

            if end >= len(text):
                break

            # 2. Compute next start with overlap, snapping forward to a whole-word boundary
            next_start = max(start + 20, end - effective_overlap)
            if next_start > 0 and not text[next_start - 1].isspace():
                next_space = text.find(" ", next_start, end)
                if next_space != -1:
                    next_start = next_space + 1
            start = next_start

    st.session_state.rag_data["chunks"] = chunks

    st.markdown("#### 🔍 Step 2 Output Inspection (Word-Aligned Chunks)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Chunks Generated", len(chunks))
    avg_chars = round(sum(c["chars"] for c in chunks) / len(chunks)) if chunks else 0
    m2.metric("Avg. Chunk Length", f"{avg_chars} chars")
    m3.metric("Boundary Mode", "Whole-Word Snapped")

    if len(chunks) >= 2 and chunks[0]["source"] == chunks[1]["source"] and effective_overlap > 0:
        shared_tail = chunks[0]["text"][-effective_overlap:]
        with st.expander("🔗 Visual Proof: How Overlap Bridges Chunk #1 and Chunk #2", expanded=True):
            c_left, c_right = st.columns(2)
            c_left.info(f"**{chunks[0]['chunk_id']} (End):**\n\n...{chunks[0]['text'][-110:]}")
            c_right.success(f"**{chunks[1]['chunk_id']} (Start):**\n\n{chunks[1]['text'][:110]}...")
            st.code(f'Shared Overlap Bridge: "{shared_tail}"', language="text")

    st.dataframe(pd.DataFrame(chunks), use_container_width=True, hide_index=True)
    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=len(chunks) > 0)

# =========================================================
# STEP 3: VECTOR EMBEDDINGS & SEMANTIC INDEX
# =========================================================
elif step == 3:
    st.subheader("Step 3: Dense Vector Embeddings & Semantic Index")
    st.write(
        "Computers cannot compare raw text meaning directly. Here, every text chunk is transformed into a "
        "**12-dimensional L2-normalized numerical vector** (unit length = `1.0`) so Step 4 can measure "
        "semantic closeness using **Cosine Similarity**."
    )

    chunks = st.session_state.rag_data.get("chunks", [])
    indexed_chunks = []

    for ch in chunks:
        vec = compute_embedding(ch["text"])
        l2_norm = round(math.sqrt(sum(v * v for v in vec)), 4)
        indexed_chunks.append(
            {
                **ch,
                "vector": vec,
                "l2_norm": l2_norm,
                "vector_preview": "[" + ", ".join(f"{v:.2f}" for v in vec[:6]) + ", ...]",
            }
        )

    st.session_state.rag_data["indexed_chunks"] = indexed_chunks

    st.markdown("#### 🔍 Step 3 Output Inspection (Vector Store Matrix)")
    v1, v2, v3 = st.columns(3)
    v1.metric("Vectors Indexed", len(indexed_chunks))
    v2.metric("Embedding Dimensions", len(CONCEPT_AXES))
    v3.metric("Distance Metric", "Cosine Similarity")

    if indexed_chunks:
        selected_id = st.selectbox(
            "Inspect the numerical vector coordinates of a specific chunk:",
            options=[item["chunk_id"] for item in indexed_chunks],
        )
        chosen = next(item for item in indexed_chunks if item["chunk_id"] == selected_id)
        st.caption(f"**Chunk Text:** {chosen['text']}")

        axis_labels = [axis[0] for axis in CONCEPT_AXES]
        vec_df = pd.DataFrame({"Semantic Dimension": axis_labels, "Activation Weight": chosen["vector"]})
        st.bar_chart(vec_df.set_index("Semantic Dimension"), height=220)

    display_df = pd.DataFrame(indexed_chunks)[["chunk_id", "source", "page", "l2_norm", "vector_preview"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=len(indexed_chunks) > 0)

# =========================================================
# STEP 4: QUERY EMBEDDING & TOP-K SEMANTIC RETRIEVAL
# =========================================================
elif step == 4:
    st.subheader("Step 4: Query Vectorization & Top-K Cosine Retrieval")
    st.write(
        "Ask a question about human behavior, personality traits, or body language. Your question is converted "
        "into the **exact same 12-dimensional vector space** and compared against every indexed chunk."
    )

    PRESET_QUESTIONS = [
        "What do pacifying behaviors like neck touching or lip compression actually indicate?",
        "How does the Fundamental Attribution Error differ from the Actor-Observer Bias?",
        "Which personality traits predict workplace reliability and ethical behavior?",
        "Custom question...",
    ]

    preset_choice = st.selectbox("Pick a sample behavioral question (or choose Custom):", PRESET_QUESTIONS)
    if preset_choice == "Custom question...":
        query = st.text_input(
            "Type your custom behavioral question:",
            value="What does ventral fronting mean in body language?",
        )
    else:
        query = st.text_input("Active Search Query:", value=preset_choice)

    col_k, col_thresh = st.columns(2)
    indexed_chunks = st.session_state.rag_data.get("indexed_chunks", [])
    max_k = max(1, min(6, len(indexed_chunks)))
    top_k = col_k.slider("Top-K Chunks to Retrieve:", min_value=1, max_value=max_k, value=min(3, max_k))
    min_score = col_thresh.slider("Minimum Cosine Similarity Threshold:", 0.0, 0.80, 0.25, 0.05)

    query_vec = compute_embedding(query)
    scored_chunks = []
    for item in indexed_chunks:
        sim = cosine_similarity(query_vec, item["vector"])
        scored_chunks.append({**item, "similarity_score": sim})

    ranked_all = sorted(scored_chunks, key=lambda x: x["similarity_score"], reverse=True)

    retrieved = [c for c in ranked_all[:top_k] if c["similarity_score"] >= min_score]
    for idx, row in enumerate(ranked_all, start=1):
        row["rank"] = idx
        row["status"] = "✅ RETRIEVED" if row in retrieved else "⚪ Filtered Out"

    st.session_state.rag_data["query"] = query
    st.session_state.rag_data["retrieved"] = retrieved

    st.markdown("#### 🔍 Step 4 Output Inspection (Similarity Ranking)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Chunks Evaluated", len(ranked_all))
    r2.metric("Chunks Passing Filter", len(retrieved))
    top_sim = ranked_all[0]["similarity_score"] if ranked_all else 0.0
    r3.metric("Highest Similarity Score", f"{top_sim:.4f}")

    with st.expander("📐 Inspect Query Embedding Vector", expanded=False):
        st.code(f"Query Vector (12-dim): {query_vec}", language="text")

    ranking_df = pd.DataFrame(ranked_all)[
        ["rank", "status", "similarity_score", "chunk_id", "source", "page", "text"]
    ]
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=bool(query.strip()))

# =========================================================
# STEP 5: CONTEXT-GROUNDED PROMPT & FINAL CITED ANSWER
# =========================================================
elif step == 5:
    st.subheader("Step 5: Context-Grounded Prompt Assembly & Final Cited Answer")
    st.write(
        "In the final stage, only the **retrieved chunks from Step 4** are injected into the LLM's system "
        "prompt with strict grounding instructions—preventing outside guesswork or hallucination."
    )

    query = st.session_state.rag_data.get("query", "")
    retrieved = st.session_state.rag_data.get("retrieved", [])

    if retrieved:
        injected_context = "\n\n".join(
            [f"[{c['source']} | {c['page']} | Score={c['similarity_score']}]\n{c['text']}" for c in retrieved]
        )
    else:
        injected_context = "[NO CHUNKS PASSED THE SIMILARITY THRESHOLD]"

    full_prompt_payload = (
        "SYSTEM INSTRUCTION:\n"
        "Answer strictly using ONLY the verified context passages below. Cite exact source filenames and page numbers.\n\n"
        f"INJECTED CONTEXT ({len(retrieved)} chunks):\n"
        f"{injected_context}\n\n"
        f"USER QUESTION:\n{query}"
    )

    with st.expander("🔍 Inspect Exact Prompt Payload Sent to Generator", expanded=True):
        st.code(full_prompt_payload, language="text")

    final_answer, telemetry = synthesize_grounded_answer(api_key, query, retrieved)

    st.markdown("#### ✅ Final Grounded RAG Output")
    t1, t2, t3 = st.columns(3)
    t1.metric("Execution Engine", telemetry["mode"])
    t2.metric("Generation Latency", f"{telemetry['latency_ms']} ms")
    t3.metric("Context + Output Tokens", telemetry["tokens_used"])

    st.success(final_answer)

    audit_report = (
        f"# GlassBox AI — Behavioral RAG Audit Report\n\n"
        f"**Question:** {query}\n\n"
        f"## Grounded Answer\n{final_answer}\n\n"
        f"## Prompt Payload & Retrieved Context\n```text\n{full_prompt_payload}\n```\n"
    )
    st.download_button(
        "⬇ Download Full RAG Trace Report (.md)",
        data=audit_report,
        file_name="behavioral_rag_audit.md",
        mime="text/markdown",
    )

    render_wizard_nav(total_steps=len(RAG_STEPS), can_proceed=True)
