"""PatchContext — Streamlit entry point.

Chat over the fastapi/fastapi development history with verified citations.
Run locally with:  streamlit run app.py
"""

import json
import logging

import streamlit as st

# Surface pipeline INFO logs (retrieval, guard decisions) to stdout so they
# appear in Cloud Run / container logs.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from patchcontext.config import REPO_ROOT, settings
from patchcontext.ui_helpers import (
    flag_unsupported,
    linkify_citations,
    load_eval_results,
    load_ref_urls,
)

st.set_page_config(page_title="PatchContext", page_icon="🔍", layout="wide")

EVAL_RESULTS_DIR = REPO_ROOT / "src" / "patchcontext" / "eval" / "results"


# --- cached resources (loaded once per process) -------------------------------

@st.cache_resource(show_spinner="Loading FAISS index and metadata …")
def get_retriever():
    from patchcontext.retrieve.retriever import Retriever

    return Retriever(settings.index_dir)


@st.cache_resource(show_spinner="Warming up local models (embedding, reranker, NLI) …")
def warm_models() -> bool:
    from patchcontext.guard.nli_guard import get_model as nli_model
    from patchcontext.index.embedder import get_model as embedding_model
    from patchcontext.retrieve.reranker import get_model as reranker_model

    embedding_model()
    reranker_model()
    nli_model()
    return True


@st.cache_resource
def get_llm():
    from patchcontext.generate.llm_client import LLMClient

    return LLMClient()


@st.cache_resource
def get_known_refs() -> set[str]:
    from patchcontext.pipeline import load_known_refs

    return load_known_refs(settings.index_dir)


@st.cache_resource
def get_ref_urls() -> dict[str, str]:
    return load_ref_urls(settings.index_dir)


@st.cache_resource
def get_index_stats() -> dict:
    stats_path = settings.index_dir / "stats.json"
    return json.loads(stats_path.read_text()) if stats_path.exists() else {}


# --- sidebar -------------------------------------------------------------------

with st.sidebar:
    st.title("PatchContext")
    st.caption(
        "Ask *why fastapi/fastapi is designed the way it is* — answers cite "
        "real commits, PRs, and issues, and every answer is checked by a "
        "hallucination guard before display."
    )
    stats = get_index_stats()
    if stats:
        st.subheader("Index")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", f"{stats.get('chunks', 0):,}")
        col2.metric("Dim", stats.get("dim", "—"))
        st.caption(
            f"History {stats.get('date_min', '?')} → {stats.get('date_max', '?')} · "
            f"built {str(stats.get('built_at', '?'))[:10]}"
        )
        by_type = stats.get("by_source_type", {})
        st.caption(" · ".join(f"{k}: {v:,}" for k, v in by_type.items()))
    st.subheader("Models")
    st.caption(f"Embedding: `{settings.embedding_model}`")
    st.caption(f"Reranker: `{settings.reranker_model}`")
    st.caption(f"Guard: `{settings.nli_model}` (τ={settings.nli_threshold})")

    llm_error = None
    try:
        llm = get_llm()
        active = llm.active or "primary (on first call)"
        st.caption(f"LLM: `{settings.llm_model}` via **{active}**")
    except ValueError as exc:
        llm, llm_error = None, str(exc)
        st.error(f"LLM not configured: {llm_error}")


# --- tabs ------------------------------------------------------------------------

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])


def render_answer(entry: dict) -> None:
    answer = entry["answer"]
    guard = answer.guard
    ref_urls = get_ref_urls()

    if guard.verdict == "verified":
        badge = "✅ verified" + (" · regenerated once" if guard.regenerated else "")
        st.caption(badge + f" · answered by {answer.provider}:{answer.model}")
        text = answer.text
    else:
        st.caption(f"⚠️ flagged · answered by {answer.provider}:{answer.model}")
        st.warning(
            "The guard could not verify parts of this answer: "
            f"{guard.failure_reason or 'unsupported claims'} — flagged claims "
            "are marked ⚠️ below."
        )
        text = flag_unsupported(answer.text, guard.nli_result.unsupported_claims)

    st.markdown(linkify_citations(text, ref_urls))

    with st.expander(f"Retrieved chunks (top {len(answer.chunks)})"):
        for i, chunk in enumerate(answer.chunks, 1):
            m = chunk.metadata
            ref = {"pr": f"PR #{m['ref_id']}", "issue": f"Issue #{m['ref_id']}"}.get(
                m["source_type"], f"commit {m['ref_id']}"
            )
            st.markdown(
                f"**{i}. [{ref}]({m['url']})** · {m['title'][:90]}  \n"
                f"score `{chunk.score:+.3f}` · {m['section']} · {m['author']} · {str(m['date'])[:10]}"
            )
            snippet = " ".join(chunk.text.split())
            st.caption(snippet[:500] + ("…" if len(snippet) > 500 else ""))


with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    # Warm the index + models at page load, not on the first query: loading
    # ~2.4 GB inside a query blocks long enough that the websocket resets and
    # the question is lost. Paying the cost at render keeps queries reliable.
    if llm is not None:
        get_retriever()
        warm_models()
        get_known_refs()

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry)

    question = st.chat_input(
        "e.g. Why was pydantic v2 adopted?", disabled=llm is None
    )
    if llm is None:
        st.info("Set an LLM key in .env (see .env.example) to enable chat.")
    if question and llm is not None:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            from patchcontext.pipeline import answer_question

            warm_models()
            with st.spinner("Retrieving, generating, and verifying …"):
                answer = answer_question(question, get_retriever(), llm, get_known_refs())
            entry = {"question": question, "answer": answer}
            st.session_state.history.append(entry)
            render_answer(entry)


with tab_eval:
    st.subheader("RAGAs evaluation")
    results = load_eval_results(EVAL_RESULTS_DIR)
    if not results:
        st.info(
            "No evaluation results yet — the 50-question RAGAs benchmark runs "
            "and its results will render here."
        )
    else:
        for result in results:
            st.markdown(f"**{result['name']}** · judge: `{result.get('judge_model', '?')}` · "
                        f"{result.get('n_questions', '?')} questions · {str(result.get('run_at', ''))[:19]}")
            metrics = result["metrics"]
            st.dataframe(
                {"metric": list(metrics.keys()), "score": [round(v, 4) for v in metrics.values()]},
                hide_index=True,
            )
            answerable = result.get("metrics_answerable_only")
            if answerable:
                st.caption(
                    f"answer_relevancy on the {answerable['n']} answerable questions "
                    f"(excluding correctly-refused unanswerables): "
                    f"**{answerable['answer_relevancy']:.3f}**"
                )
            if result.get("analysis"):
                st.info(result["analysis"])
        st.caption(
            "**Methodology:** each benchmark question runs through the full "
            "retrieve → rerank → generate → guard pipeline; RAGAs scores the "
            "answers with an independent judge LLM (configured via "
            "`RAGAS_JUDGE_*`). The benchmark mixes direct, multi-hop "
            "(PR↔issue), and deliberately unanswerable questions."
        )
