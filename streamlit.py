import time
import streamlit as st
from main import run_research_pipeline, save_output_to_file, topic_to_filename

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research & Critic Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling - keep it clean, not templated. A single accent color,
# card-like containers, calmer typography than Streamlit's defaults.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --accent: #2563EB;
            --accent-soft: #EFF4FF;
            --ink: #14213D;
            --muted: #5B6472;
        }

        .main .block-container {
            padding-top: 2rem;
            max-width: 1100px;
        }

        h1, h2, h3 {
            color: var(--ink);
            font-weight: 700;
        }

        .app-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            margin-bottom: 1.6rem;
        }

        .pipeline-step {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 999px;
            padding: 0.25rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        .source-card {
            border: 1px solid #E4E7EC;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
            background: #FAFBFC;
        }

        .source-title {
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 0.15rem;
        }

        .source-url {
            font-size: 0.82rem;
            color: var(--muted);
            word-break: break-all;
        }

        .stButton>button {
            background: var(--accent);
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
        }

        .stButton>button:hover {
            background: #1D4FD1;
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><h1>🔎 AI Research &amp; Critic Agent</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Searches the web, scrapes real sources, '
    "writes a grounded report, then critiques its own work for accuracy.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="pipeline-step">1. Search (Tavily)</span>'
    '<span class="pipeline-step">2. Scrape (BeautifulSoup)</span>'
    '<span class="pipeline-step">3. Writer Chain (Gemini)</span>'
    '<span class="pipeline-step">4. Critic Chain (Gemini)</span>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar - inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. cybersecurity challenges in the AI era",
    )
    max_results = st.slider(
        "Number of sources to search",
        min_value=3,
        max_value=10,
        value=5,
        help="More sources = more thorough report, but slower and more API usage.",
    )
    save_to_disk = st.checkbox("Also save result as a .txt file", value=True)
    run_clicked = st.button("🚀 Run Research Pipeline", use_container_width=True)

    st.divider()
    st.caption(
        "Powered by LangChain + Google Gemini + Tavily. "
        "Make sure GOOGLE_API_KEY and TAVILY_API_KEY are set in your .env file."
    )

# ---------------------------------------------------------------------------
# Session state - keep last result around across reruns
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.topic = None

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        with st.spinner(""):
            steps = [
                "🔍 Searching the web for relevant sources...",
                "📄 Reading and scraping page content...",
                "✍️ Writer chain is drafting the report...",
                "🧐 Critic chain is reviewing the draft...",
            ]
            # Lightweight visual progress while the real pipeline runs.
            # (The pipeline itself is synchronous, so this just gives
            # the user a sense of where things likely are.)
            status_placeholder.info(steps[0])
            progress_bar = progress_placeholder.progress(10)

            try:
                result = run_research_pipeline(topic.strip(), max_results=max_results)
                progress_bar.progress(100)
                status_placeholder.success("Done!")
                time.sleep(0.4)
            except Exception as e:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error(f"Something went wrong while running the pipeline:\n\n{e}")
                result = None

        progress_placeholder.empty()
        status_placeholder.empty()

        if result is not None:
            st.session_state.result = result
            st.session_state.topic = topic.strip()

            if save_to_disk:
                saved_path = save_output_to_file(topic.strip(), result["final_output"])
                st.session_state.saved_path = saved_path

# ---------------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.info("👈 Enter a topic in the sidebar and click **Run Research Pipeline** to get started.")
else:
    if not result["sources"]:
        st.warning(result["final_output"])
    else:
        st.subheader(f"Results for: *{st.session_state.topic}*")

        col1, col2, col3 = st.columns(3)
        col1.metric("Sources found", len(result["sources"]))
        col2.metric("Report length", f"{len(result['report'].split())} words")
        col3.metric("Critique length", f"{len(result['critique'].split())} words")

        if save_to_disk and "saved_path" in st.session_state:
            st.caption(f"💾 Saved to: `{st.session_state.saved_path}`")

        tab_report, tab_critique, tab_sources, tab_combined = st.tabs(
            ["📝 Report", "🧐 Critic Review", "🔗 Sources", "📋 Combined Output"]
        )

        with tab_report:
            st.markdown(result["report"])
            st.download_button(
                "⬇️ Download Report (.txt)",
                data=result["report"],
                file_name=f"report_{topic_to_filename(st.session_state.topic)}",
                mime="text/plain",
            )

        with tab_critique:
            st.markdown(result["critique"])
            st.download_button(
                "⬇️ Download Critique (.txt)",
                data=result["critique"],
                file_name=f"critique_{topic_to_filename(st.session_state.topic)}",
                mime="text/plain",
            )

        with tab_sources:
            for i, s in enumerate(result["sources"], start=1):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">[{i}] {s['title']}</div>
                        <div class="source-url">{s['url']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("View scraped content"):
                    st.text(s["full_content"][:2000] + (" ...[truncated]" if len(s["full_content"]) > 2000 else ""))

        with tab_combined:
            st.text(result["final_output"])
            st.download_button(
                "⬇️ Download Full Output (.txt)",
                data=result["final_output"],
                file_name=topic_to_filename(st.session_state.topic),
                mime="text/plain",
                use_container_width=True,
            )
