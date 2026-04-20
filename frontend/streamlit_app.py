import html
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from frontend.api_client import (
    BackendClientError,
    answer_question,
    get_health,
    get_index_status,
    reset_index,
    upload_document,
)


settings = get_settings()


def _resolve_backend_base_url() -> str:
    secret_backend_url = st.secrets.get("BACKEND_BASE_URL", "")
    if isinstance(secret_backend_url, str) and secret_backend_url.strip():
        return secret_backend_url.strip()
    return settings.backend_base_url

st.set_page_config(
    page_title=settings.app_name,
    page_icon=":page_facing_up:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.session_state.setdefault("upload_response", None)
st.session_state.setdefault("answer_response", None)
st.session_state.setdefault("flash_notice", None)
st.session_state.setdefault("uploader_nonce", 0)
st.session_state.setdefault("question_input", "")
st.session_state.setdefault("clear_question_pending", False)

if st.session_state["clear_question_pending"]:
    st.session_state["question_input"] = ""
    st.session_state["clear_question_pending"] = False


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --surface: rgba(255, 252, 247, 0.84);
            --text: #182320;
            --muted: #60706a;
            --line: rgba(24, 58, 50, 0.10);
            --teal: #0f766e;
            --teal-deep: #134e4a;
            --shadow: 0 18px 40px rgba(16, 24, 40, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(217, 119, 6, 0.10), transparent 24%),
                linear-gradient(180deg, #faf5ef 0%, #f5efe6 100%);
            color: var(--text);
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            color: var(--text);
            letter-spacing: -0.03em;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(19, 78, 74, 0.98), rgba(14, 63, 61, 0.98));
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f4f7f6;
        }

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f5fbfa;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 26px;
            box-shadow: var(--shadow);
            padding: 0.25rem;
            backdrop-filter: blur(10px);
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(24, 58, 50, 0.08);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            color: var(--text);
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            border-radius: 999px;
            border: 1px solid rgba(24, 58, 50, 0.10);
            background: rgba(255, 255, 255, 0.72);
            color: var(--text);
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            font-weight: 800;
            letter-spacing: -0.01em;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            border: 0;
            background: linear-gradient(135deg, var(--teal), #16958d);
            color: #f7fffd;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.22);
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
        }

        .stTextInput input,
        .stNumberInput input {
            border-radius: 16px;
            border: 1px solid rgba(24, 58, 50, 0.12);
            background: rgba(255, 255, 255, 0.92);
        }

        [data-testid="stFileUploader"] section {
            border-radius: 22px;
            border: 1.5px dashed rgba(15, 118, 110, 0.26);
            background:
                linear-gradient(180deg, rgba(15, 118, 110, 0.05), rgba(255, 255, 255, 0.9));
            padding: 1rem;
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid rgba(24, 58, 50, 0.08);
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.95rem 2rem 2rem 2rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.22), transparent 28%),
                linear-gradient(135deg, #123f3d 0%, #176861 58%, #d07b17 130%);
            color: #f7faf8;
            box-shadow: 0 28px 56px rgba(18, 63, 61, 0.20);
            margin-bottom: 1.25rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            right: -40px;
            bottom: -50px;
            width: 230px;
            height: 230px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.10);
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            color: #e5f2f0;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0.8rem 0 0 0;
            font-size: clamp(2.1rem, 3.8vw, 3.5rem);
            line-height: 0.98;
            color: #fffdf9;
        }

        .hero-subtitle {
            max-width: 38rem;
            margin-top: 0.75rem;
            color: rgba(247, 250, 248, 0.88);
            font-size: 0.98rem;
            line-height: 1.48;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.13);
            color: #f7faf8;
            font-size: 0.9rem;
            font-weight: 700;
        }

        .hero-badge::before {
            content: "";
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            display: inline-block;
        }

        .badge-ready::before {
            background: #8ae0b3;
        }

        .badge-pending::before {
            background: #f8c861;
        }

        .badge-offline::before {
            background: #fca5a5;
        }

        .step-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1.15rem;
        }

        .step-pill {
            padding: 0.95rem 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.11);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .step-label {
            color: rgba(247, 250, 248, 0.74);
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .step-text {
            margin-top: 0.32rem;
            color: #fffef9;
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.3;
        }

        .section-kicker {
            color: var(--teal);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }

        .section-title {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -0.03em;
            margin-bottom: 0.22rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.42;
            margin-bottom: 0.7rem;
        }

        .micro-note {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            color: var(--teal-deep);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .empty-card {
            border-radius: 20px;
            padding: 1rem 1.05rem;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(24, 58, 50, 0.08);
            color: var(--text);
            line-height: 1.5;
        }

        .answer-card {
            border-radius: 26px;
            padding: 1.35rem 1.45rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(248, 253, 251, 0.92));
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: 0 16px 32px rgba(16, 24, 40, 0.05);
        }

        .answer-label {
            color: var(--teal);
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .answer-text {
            margin-top: 0.55rem;
            color: var(--text);
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            font-size: 1.28rem;
            font-weight: 700;
            line-height: 1.55;
        }

        .fallback-card {
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            background: rgba(255, 249, 237, 0.96);
            border: 1px solid rgba(217, 119, 6, 0.18);
        }

        .fallback-title {
            color: #8a4b04;
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            font-size: 1rem;
            font-weight: 800;
        }

        .fallback-copy {
            margin-top: 0.42rem;
            color: #8a4b04;
            line-height: 1.5;
        }

        .source-card {
            border-radius: 22px;
            padding: 1rem 1.05rem 1.08rem 1.05rem;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(24, 58, 50, 0.08);
        }

        .source-head {
            display: flex;
            justify-content: space-between;
            gap: 0.7rem;
            align-items: flex-start;
        }

        .source-file {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            font-size: 1rem;
            font-weight: 800;
            color: var(--text);
        }

        .source-meta {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.4rem;
        }

        .meta-chip {
            padding: 0.35rem 0.62rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            color: var(--teal-deep);
            font-size: 0.76rem;
            font-weight: 700;
        }

        .source-text {
            margin-top: 0.78rem;
            color: #24342f;
            font-size: 0.97rem;
            line-height: 1.67;
        }

        .source-caption {
            margin-top: 0.5rem;
            color: var(--muted);
            font-size: 0.8rem;
        }

        .utility-card {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(24, 58, 50, 0.08);
        }

        .utility-kicker {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }

        .utility-title {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .utility-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0.75rem;
        }

        @media (max-width: 900px) {
            .step-strip {
                grid-template-columns: 1fr;
            }

            .source-head {
                flex-direction: column;
            }

            .source-meta {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_flash_notice() -> None:
    notice = st.session_state.get("flash_notice")
    if not notice:
        return

    kind = notice.get("kind", "info")
    message = notice.get("message", "")
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)

    st.session_state["flash_notice"] = None


def _friendly_error(action: str, exc: BackendClientError) -> str:
    if action == "status":
        return f"Couldn't check the app status. {exc.message}"
    if action == "upload":
        return f"Your file could not be uploaded. {exc.message}"
    if action == "answer":
        return f"The app could not create an answer right now. {exc.message}"
    if action == "reset":
        return f"The app could not clear your current document."
    return exc.message


def _section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{html.escape(kicker)}</div>
        <div class="section-title">{html.escape(title)}</div>
        <div class="section-copy">{html.escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def _utility_header(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="utility-kicker">Utility</div>
        <div class="utility-title">{html.escape(title)}</div>
        <div class="utility-copy">{html.escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(*, backend_ready: bool, index_status: dict | None) -> None:
    if backend_ready and index_status and index_status.get("indexed"):
        ready_badge = '<span class="hero-badge badge-ready">System ready</span>'
        docs_badge = '<span class="hero-badge badge-ready">Document loaded</span>'
    elif backend_ready:
        ready_badge = '<span class="hero-badge badge-ready">System ready</span>'
        docs_badge = '<span class="hero-badge badge-pending">Upload a document to begin</span>'
    else:
        ready_badge = '<span class="hero-badge badge-offline">Connection needed</span>'
        docs_badge = '<span class="hero-badge badge-offline">App unavailable</span>'

    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">Answer questions from your own files</div>
            <h1>{html.escape(settings.app_name)}</h1>
            <div class="hero-subtitle">
                Upload a PDF or TXT file, ask a question, and get an answer based only on that document.
            </div>
            <div class="hero-badges">
                {ready_badge}
                {docs_badge}
            </div>
            <div class="step-strip">
                <div class="step-pill">
                    <div class="step-label">1</div>
                    <div class="step-text">Upload your document.</div>
                </div>
                <div class="step-pill">
                    <div class="step-label">2</div>
                    <div class="step-text">Ask a question.</div>
                </div>
                <div class="step-pill">
                    <div class="step-label">3</div>
                    <div class="step-text">Read the answer and excerpts.</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_source_card(source: dict) -> None:
    page_label = f"Page {source['page_number']}" if source["page_number"] is not None else "TXT file"
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-head">
                <div class="source-file">{html.escape(source['filename'])}</div>
                <div class="source-meta">
                    <span class="meta-chip">Excerpt {source['rank']}</span>
                    <span class="meta-chip">{html.escape(page_label)}</span>
                </div>
            </div>
            <div class="source-text">{html.escape(source['snippet'])}</div>
            <div class="source-caption">Section ID: {html.escape(source['chunk_id'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_styles()

st.sidebar.markdown("## Settings")
st.sidebar.caption("Hidden by default.")
with st.sidebar.expander("Advanced options", expanded=False):
    backend_url = st.text_input(
        "App URL",
        value=_resolve_backend_base_url(),
        help="Address of the local app service.",
    ).strip()
    top_k = st.number_input(
        "Answer detail",
        min_value=1,
        max_value=10,
        value=settings.retrieval_top_k,
        step=1,
        help="How many prepared sections the app checks when building an answer.",
    )

if "backend_url" not in locals():
    backend_url = _resolve_backend_base_url()
if "top_k" not in locals():
    top_k = settings.retrieval_top_k

try:
    get_health(base_url=backend_url)
    backend_ready = True
except BackendClientError as exc:
    backend_ready = False
    st.error(exc.message)
    st.info("Start the local app first, then refresh this page or update the app URL in the sidebar.")

index_status = None
index_status_error = None
if backend_ready:
    try:
        index_status = get_index_status(base_url=backend_url)
    except BackendClientError as exc:
        index_status_error = _friendly_error("status", exc)

_render_hero(backend_ready=backend_ready, index_status=index_status)
_show_flash_notice()

with st.container(border=True):
    _section_header("Upload", "Add your document", "PDF or TXT")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        key=f"uploader-{st.session_state['uploader_nonce']}",
        help="The app reads the file and prepares it for question answering.",
    )
    upload_clicked = st.button(
        "Upload and prepare document",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or not backend_ready,
    )

    if upload_clicked and uploaded_file is not None:
        with st.spinner("Uploading and preparing your document..."):
            try:
                upload_response = upload_document(
                    base_url=backend_url,
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    content_type=uploaded_file.type or "application/octet-stream",
                )
                st.session_state["upload_response"] = upload_response
                st.session_state["answer_response"] = None
                st.session_state["flash_notice"] = {
                    "kind": "success",
                    "message": f"Uploaded and prepared `{upload_response['filename']}`.",
                }
                st.rerun()
            except BackendClientError as exc:
                st.error(_friendly_error("upload", exc))

    upload_response = st.session_state["upload_response"]
    if upload_response:
        upload_metrics = st.columns(2)
        upload_metrics[0].metric("Prepared sections", upload_response["chunk_count"])
        upload_metrics[1].metric("File type", upload_response["source_type"].upper())
    else:
        st.markdown(
            """
            <div class="empty-card">
                Upload a file to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    _section_header("Question", "Ask about the document", "Type your question below.")
    index_ready = bool(index_status and index_status.get("indexed"))
    if backend_ready and not index_ready:
        st.markdown(
            """
            <div class="empty-card">
                Upload a document first.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("question_form"):
        st.text_input(
            "Question",
            placeholder="What file types does the system accept?",
            key="question_input",
            disabled=not backend_ready or not index_ready,
        )
        ask_clicked = st.form_submit_button(
            "Ask about the document",
            type="primary",
            use_container_width=True,
            disabled=not backend_ready or not index_ready,
        )

    if ask_clicked:
        normalized_question = st.session_state["question_input"].strip()
        if not normalized_question:
            st.warning("Type a question before asking.")
        else:
            with st.spinner("Finding the answer in your document..."):
                try:
                    st.session_state["answer_response"] = answer_question(
                        base_url=backend_url,
                        question=normalized_question,
                        top_k=int(top_k),
                    )
                except BackendClientError as exc:
                    st.session_state["answer_response"] = None
                    st.error(_friendly_error("answer", exc))

answer_response = st.session_state["answer_response"]

with st.container(border=True):
    _section_header("Answer", "Your answer", "Based only on your document.")
    if answer_response:
        if answer_response["answer_supported"]:
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-label">Your answer</div>
                    <div class="answer-text">{html.escape(answer_response['answer'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="fallback-card">
                    <div class="fallback-title">The answer was not found in your document</div>
                    <div class="fallback-copy">
                        Try a simpler question or upload a file that covers it directly.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if answer_response["sources"]:
            st.markdown(
                """
                <div style="height: 0.45rem;"></div>
                <div class="section-kicker">Supporting excerpts</div>
                <div class="section-title" style="font-size: 1.18rem; margin-bottom: 0.18rem;">Supporting excerpts</div>
                """,
                unsafe_allow_html=True,
            )
            for source in answer_response["sources"]:
                _render_source_card(source)
                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="empty-card">
                Your answer will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )

utility_left, utility_right = st.columns([1.45, 0.85], gap="large")

with utility_left:
    with st.container(border=True):
        _utility_header("Your document", "Currently loaded.")
        if index_status_error:
            st.warning(index_status_error)
        elif not index_status or not index_status["indexed"]:
            st.markdown(
                """
                <div class="utility-card">
                    No document loaded.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            metrics = st.columns(2)
            metrics[0].metric("Uploaded documents", index_status["document_count"])
            metrics[1].metric("Prepared sections", index_status["chunk_count"])
            doc_pills = "".join(
                f'<span class="pill">{html.escape(name)}</span>'
                for name in index_status["filenames"]
            )
            st.markdown(
                f"""
                <div class="utility-card">
                    <div class="micro-note">Ready to answer from:</div>
                    <div class="pill-row">{doc_pills}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if upload_response:
            st.markdown(
                f"""
                <div class="micro-note" style="margin-top: 0.85rem;">
                    Latest upload: <strong>{html.escape(upload_response['filename'])}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

with utility_right:
    with st.container(border=True):
        _utility_header("Start over", "Clear the current document.")
        confirm_reset = st.checkbox(
            "Clear the current document set",
            value=False,
            disabled=not backend_ready,
        )
        reset_clicked = st.button(
            "Reset",
            use_container_width=True,
            disabled=not backend_ready,
        )
        st.markdown(
            """
            <div class="micro-note">
                Use this when you want a clean slate.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if reset_clicked:
            if not confirm_reset:
                st.warning("Tick the checkbox first if you want to clear the current document set.")
            else:
                try:
                    reset_response = reset_index(base_url=backend_url)
                    st.session_state["upload_response"] = None
                    st.session_state["answer_response"] = None
                    st.session_state["clear_question_pending"] = True
                    st.session_state["uploader_nonce"] += 1
                    st.session_state["flash_notice"] = {
                        "kind": "success",
                        "message": reset_response["message"],
                    }
                    st.rerun()
                except BackendClientError as exc:
                    st.error(_friendly_error("reset", exc))
