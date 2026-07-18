"""System and answer prompts, citation format.

Requirements:
- Inline citations in exactly these formats: ``[PR #1276]``, ``[Issue #4433]``,
  ``[commit ab12cd3]``.
- If the retrieved context does not contain the answer, the model must say so
  explicitly instead of guessing — refusal is a valid answer.
- Generation parameters (temperature, max_tokens) live in config.py / LLMClient.
"""

from __future__ import annotations

from patchcontext.retrieve.retriever import RetrievedChunk

SYSTEM_PROMPT: str = """\
You are PatchContext, an assistant that answers questions about the design and
history of the fastapi/fastapi project using ONLY the provided context chunks,
which are drawn from real commits, pull requests, and issues.

Rules:
- Cite every factual claim inline using exactly one of these formats:
  [PR #1276], [Issue #4433], [commit ab12cd3]
  (the numbers/shas above are FORMAT EXAMPLES ONLY — never cite them unless
  they actually appear in the provided context)
- Cite only references that appear in the provided context chunks. Never invent
  a PR, issue, or commit reference.
- Output only the final answer — no reasoning, meta-commentary, or notes about
  these instructions. Keep it under ~300 words.
- Use ONLY the bracket citation formats above — no numeric footnotes like [1].
- State facts directly, one claim per sentence, close to the wording of the
  context. Write "Lifespan superseded startup/shutdown events [commit abc1234]",
  NOT narration about sources like "the commit that added this explicitly
  states..." or "as described in the PR...".
- If the context does not contain the answer, say so explicitly — for example:
  "The retrieved development history does not contain the answer to this."
  Refusing to answer is always better than guessing.
- Be concise and specific: quote or paraphrase what developers actually said
  or did, and attribute design decisions to the references where they happened.
"""

REGENERATION_SUFFIX: str = """\

IMPORTANT — your previous answer failed verification:
{failure_reason}

Rewrite the answer using ONLY claims that are directly supported by the context
chunks above, with a citation on every claim. Drop any claim you cannot support.
If the context does not actually contain the answer, say so explicitly.
"""


def format_ref(source_type: str, ref_id: str) -> str:
    """Human/citation form of a chunk reference: PR #N, Issue #N, commit sha."""
    if source_type == "pr":
        return f"PR #{ref_id}"
    if source_type == "issue":
        return f"Issue #{ref_id}"
    return f"commit {ref_id}"


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format the user message: numbered context chunks with their refs, then the question."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk.metadata
        ref = format_ref(m["source_type"], m["ref_id"])
        blocks.append(
            f"[{i}] {ref} — {m['title']} — {m['author']}, {str(m['date'])[:10]}\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(blocks)
    return (
        f"Context chunks from the fastapi/fastapi development history:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with an inline citation "
        "([PR #N] / [Issue #N] / [commit sha]) on every factual claim."
    )
