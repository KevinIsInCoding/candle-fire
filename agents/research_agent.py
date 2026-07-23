"""Multi-step ALS research synthesis agent (streaming). Mirrors beacon/agents/research.py pattern."""
from __future__ import annotations

import json
import re
from collections.abc import Generator

import anthropic
import chromadb
import networkx as nx
from sentence_transformers import CrossEncoder

from config import (
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_N,
    RETRIEVAL_ENTITY_N,
    RETRIEVAL_SEMANTIC_N,
    RRF_K,
    RRF_TOP_N,
    SYNTHESIS_MODEL,
)
from graph import query as kg_query
from llm import cached_system, cached_tools
from logging_config import get_logger
from normalization.drug_vocab import build_drug_vocab, suggest_drug_term
from prompts import SYNTHESIS_SYSTEM
from rag import retriever as rag_retriever
from tools import RESEARCH_TOOLS

_logger = get_logger("agents.research_agent")

# Loaded once at startup — ~80MB model, ~80ms/pair on CPU
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# Drug vocabulary is derived from the (startup-loaded) trials + graph; cache by identity
# so it is built once per session rather than on every query.
_DRUG_VOCAB_CACHE: dict[int, dict] = {}


def _get_drug_vocab(trials: list[dict], graph: nx.DiGraph | None) -> dict:
    key = id(trials)
    vocab = _DRUG_VOCAB_CACHE.get(key)
    if vocab is None:
        vocab = build_drug_vocab(trials, graph)
        _DRUG_VOCAB_CACHE.clear()  # session has one trials object; avoid unbounded growth
        _DRUG_VOCAB_CACHE[key] = vocab
    return vocab


def stream_research_agent(
    client: anthropic.Anthropic,
    query: str,
    collection: chromadb.Collection,
    trials: list[dict],
    graph: nx.DiGraph | None = None,
) -> Generator[tuple[str, str], None, None]:
    """
    Stream the ALS research synthesis agent.

    Yields:
        ("token", str)   — partial text chunk for streaming display
        ("status", str)  — status message during tool execution
        ("done", str)    — final complete response text
    """
    messages: list[anthropic.types.MessageParam] = [
        {"role": "user", "content": query}
    ]
    # Attach graph reference so _handle_search can use KG expansion
    _graph = graph
    first_turn = True

    while True:
        stream_text = ""

        # Force tool use on the first turn so Claude always searches before synthesizing.
        # Unknown proper nouns (drug codes, gene IDs) would otherwise trigger a
        # "I don't recognize X" response straight from training knowledge.
        tool_choice: dict = {"type": "any"} if first_turn else {"type": "auto"}

        with client.messages.stream(
            model=SYNTHESIS_MODEL,
            max_tokens=4096,
            system=cached_system(SYNTHESIS_SYSTEM),
            tools=cached_tools(RESEARCH_TOOLS),
            tool_choice=tool_choice,
            messages=messages,
        ) as stream:
            # Accumulate tool-use input JSON alongside streaming text
            tool_calls: list[dict] = []
            current_tool: dict | None = None
            current_input_json = ""

            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                        }
                        current_input_json = ""
                        yield ("status", "Searching knowledge base and re-ranking results for precision...")

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        chunk = event.delta.text
                        stream_text += chunk
                        yield ("token", chunk)
                    elif event.delta.type == "input_json_delta" and current_tool:
                        current_input_json += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            current_tool["input"] = json.loads(current_input_json)
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_calls.append(current_tool)
                        current_tool = None
                        current_input_json = ""

            final_msg = stream.get_final_message()

        messages.append({"role": "assistant", "content": final_msg.content})

        first_turn = False

        if final_msg.stop_reason == "end_turn":
            yield ("done", stream_text)
            return

        if final_msg.stop_reason == "tool_use" and tool_calls:
            tool_results: list[anthropic.types.ToolResultBlockParam] = []

            for tool_call in tool_calls:
                if tool_call["name"] == "search_research_landscape":
                    result = _handle_search(tool_call["input"], collection, trials, _graph)
                    is_error = False
                else:
                    result = {"error": f"Unknown tool: {tool_call['name']}"}
                    is_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": json.dumps(result),
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            yield ("done", stream_text)
            return


# Ubiquitous ALS disease descriptors — grounded across the whole corpus, so they
# must never count as a query "focus" entity for the grounding gate.
_GENERIC_EXACT = {"als", "mnd", "mnds", "ftd", "als/ftd", "disease", "neurodegeneration", "therapy", "treatment"}
_GENERIC_SUBSTRINGS = ("amyotrophic", "lateral sclerosis", "motor neuron")


def _is_generic_term(term: str) -> bool:
    """True for disease-generic terms that are grounded everywhere (ALS, MND, etc.)."""
    t = term.lower().strip()
    if t in _GENERIC_EXACT:
        return True
    return any(sub in t for sub in _GENERIC_SUBSTRINGS)


def _norm_alnum(s: str) -> str:
    """Lowercase, alphanumeric-only form so 'CNM-Au8', 'CNMAu8', 'cnm_au8' all unify."""
    return "".join(c for c in s.lower() if c.isalnum())


# Trial status display order — available/recruiting first, closed/unavailable last.
# Expanded Access uses AVAILABLE / TEMPORARILY_NOT_AVAILABLE / NO_LONGER_AVAILABLE.
_TRIAL_STATUS_RANK = {
    "AVAILABLE": 0,
    "RECRUITING": 1,
    "NOT_YET_RECRUITING": 2,
    "ENROLLING_BY_INVITATION": 3,
    "ACTIVE_NOT_RECRUITING": 4,
    "TEMPORARILY_NOT_AVAILABLE": 5,
    "COMPLETED": 6,
    "SUSPENDED": 7,
    "TERMINATED": 8,
    "WITHDRAWN": 9,
    "NO_LONGER_AVAILABLE": 10,
}


def _term_excerpt(document: str, terms: list[str], window: int = 600) -> str:
    """Return a 600-char excerpt centred on the first keyword match, or the document start."""
    doc_lower = document.lower()
    for term in terms:
        idx = doc_lower.find(term.lower())
        if idx != -1:
            start = max(0, idx - 200)
            return document[start : start + window]
    return document[:window]


def _handle_search(
    tool_input: dict,
    collection: chromadb.Collection,
    trials: list[dict],
    graph: nx.DiGraph | None = None,
) -> dict:
    """Execute KG expansion → RAG search → trial lookup and return structured context."""
    query_text = tool_input.get("query_text", "")
    query_entities = tool_input.get("query_entities", [])

    # Step 1: KG expansion — surface related entities Claude didn't name explicitly
    # e.g. "tofersen" → expands to ["SOD1", "antisense oligonucleotide", "RNA splicing"]
    if graph and query_entities:
        expanded_entities = kg_query.expand_query_entities(graph, query_entities)
    else:
        expanded_entities = query_entities

    # Step 2: Semantic search → up to 30 papers (pure similarity, no citation weight yet)
    semantic_results = rag_retriever.search(collection, query_text, n_results=RETRIEVAL_SEMANTIC_N)

    # Step 3: Entity-targeted search → up to 30 papers (one query per expanded entity)
    entity_results = rag_retriever.search_by_entities(
        collection, expanded_entities, n_results=RETRIEVAL_ENTITY_N
    )

    # Step 3b: Keyword search — exact $contains match for specific named terms.
    # Also extracts alphanumeric tokens from query_text (e.g. "SPG302", "C9orf72",
    # "AMX0035") that Claude may not include in query_entities because it doesn't
    # recognize them as known biological entities.
    _entity_tokens = list({
        tok for tok in re.findall(r'\b[A-Za-z]+\d+\w*|\b[A-Z]{2,}\d*\w*', query_text)
        if len(tok) >= 3
    })
    keyword_terms = list(dict.fromkeys(query_entities + _entity_tokens))  # dedup, preserve order
    keyword_results = rag_retriever.search_by_keyword(collection, keyword_terms)

    # Grounding gate — determine whether the query's *specific* focus entities are
    # genuinely present in the PAPER corpus. Semantic search always returns nearest
    # neighbors regardless of relevance, so we check exact literal presence
    # ($contains); otherwise Claude grafts real PMIDs onto topically-adjacent-but-
    # unrelated papers. Two exclusions from the focus set:
    #   - Ubiquitous disease descriptors (ALS / motor neuron disease) — grounded
    #     everywhere, never the subject of the query.
    #   - KG-node existence is deliberately NOT used as grounding: a compound can
    #     have a graph node purely from trial data while having zero paper evidence
    #     (e.g. SPG302). "In the corpus" means "written in a paper".
    focus_terms = [t for t in keyword_terms if t.strip() and not _is_generic_term(t)]

    # "Did you mean?" — for unrecognized focus terms, suggest the nearest known drug name
    # (typo tolerance). SUGGESTION ONLY — the original term is still what gets searched, so a
    # wrong suggestion can never silently redirect the query onto a different drug.
    drug_vocab = _get_drug_vocab(trials, graph)
    did_you_mean = {
        t: s for t in focus_terms if (s := suggest_drug_term(t, drug_vocab))
    }

    grounded_terms: list[str] = []
    ungrounded_terms: list[str] = []
    for term in focus_terms:
        if rag_retriever.is_grounded_in_corpus(collection, term):
            grounded_terms.append(term)
        else:
            ungrounded_terms.append(term)

    # If the query names specific entities and NONE are grounded in papers, the
    # corpus holds no genuine evidence — Claude gets zero papers so it cannot graft.
    evidence_ungrounded = bool(focus_terms) and not grounded_terms

    # Step 4: RRF merge → top 20 papers
    merged = rag_retriever.rrf_merge(
        [semantic_results, entity_results, keyword_results], k=RRF_K, top_n=RRF_TOP_N
    )

    # Step 5: Cross-encoder rerank → top 15 papers
    reranked = rag_retriever.cross_encoder_rerank(
        _cross_encoder, query_text, merged, top_n=CROSS_ENCODER_TOP_N
    )

    # Step 6: Citation boost — final score = ce_score × log(citation_count + 2)
    top_papers = rag_retriever.apply_citation_boost(reranked)

    # Apply the grounding gate: suppress spurious semantic matches when the query's
    # focus entity is absent from the corpus. Trials are still returned below.
    paper_pool = [] if evidence_ungrounded else top_papers

    # Guarantee the papers that literally name a *landscape-only* focus compound are
    # citable. When a compound appears only in full-text pipeline tables (no abstract
    # anywhere — e.g. SPG302), the cross-encoder ranks those table chunks below generic
    # semantic neighbors and they never reach the model, so it can neither cite nor
    # label them. Inject their keyword-hits (capped). Well-grounded entities (C9orf72,
    # tofersen) already surface primary papers via semantic/CE — skip injection for them.
    landscape_only_terms = [
        t for t in grounded_terms
        if not rag_retriever.is_grounded_in_abstract(collection, t)
    ]
    if not evidence_ungrounded and landscape_only_terms:
        present = {r["pmid"] for r in paper_pool}
        focus_hits = rag_retriever.search_by_keyword(collection, landscape_only_terms)
        for r in focus_hits[:5]:
            if r["pmid"] not in present:
                r.setdefault("score", r.get("similarity", 0.0))
                paper_pool.append(r)
                present.add(r["pmid"])

    _logger.info(
        "KG+RAG+CE search",
        extra={"data": {
            "query_entities": query_entities,
            "expanded_entities": len(expanded_entities),
            "semantic_hits": len(semantic_results),
            "entity_hits": len(entity_results),
            "keyword_hits": len(keyword_results),
            "rrf_merged": len(merged),
            "after_cross_encoder": len(top_papers),
            "grounded_terms": grounded_terms,
            "ungrounded_terms": ungrounded_terms,
            "evidence_ungrounded": evidence_ungrounded,
            "papers_returned": len(paper_pool),
            "kg_active": graph is not None,
        }},
    )

    # Step 7: Trial matching — return ONLY trials genuinely about the queried compound
    # or target, ranked available/recruiting first (EAPs, completed, and terminated all
    # included). Match on the SPECIFIC query terms — never expanded_entities, whose KG
    # expansion balloons to thousands of terms and floods results with unrelated ALS
    # trials. Normalized (alphanumeric-only) matching unifies "CNM-Au8" / "CNMAu8" /
    # "cnm_au8" across the query, trial interventions, and enriched target_entities.
    specific_terms = [
        t for t in dict.fromkeys(query_entities + focus_terms)
        if t.strip() and not _is_generic_term(t)
    ]
    norm_terms = [n for n in (_norm_alnum(t) for t in specific_terms) if len(n) >= 3]
    # Drop fragments that are substrings of a longer matched term — the query regex
    # splits "CNM-Au8" into "CNM"/"Au8", whose short normalized forms ("cnm"/"au8")
    # over-match unrelated trials. Keep only maximal terms (e.g. "cnmau8").
    norm_terms = [n for n in norm_terms if not any(n != m and n in m for m in norm_terms)]
    nct_ids_in_query = {w.upper() for w in query_text.split() if w.upper().startswith("NCT")}

    matched: dict[str, dict] = {}
    for trial in trials:
        nct = trial.get("nct_id", "")
        if not nct:
            continue
        iv_names = " ".join(iv.get("name", "") for iv in trial.get("interventions", []))
        targets = " ".join(trial.get("target_entities", []))
        hay = _norm_alnum(f"{trial.get('title', '')} {iv_names} {targets} {trial.get('summary', '')}")
        if nct.upper() in nct_ids_in_query or any(nt in hay for nt in norm_terms):
            matched[nct] = trial

    ranked = sorted(matched.values(), key=lambda t: _TRIAL_STATUS_RANK.get(t.get("status", ""), 99))
    related_trials = [
        {
            "nct_id": t.get("nct_id", ""),
            "title": t.get("title", ""),
            "phase": t.get("phase", ""),
            "status": t.get("status", ""),
            "study_type": t.get("study_type", ""),
            "url": t.get("url", ""),
        }
        for t in ranked[:10]
    ]

    # Context-aware grounding note steers the synthesis model away from hallucination.
    if evidence_ungrounded:
        _terms = ", ".join(ungrounded_terms) or "the queried entity"
        grounding_note = (
            f"NO paper evidence exists in this database for: {_terms}. "
            "Do NOT synthesize a mechanism or any factual claim from training knowledge, and do "
            "NOT cite any PMID. State explicitly that the paper database contains no evidence for "
            f"{_terms}. Report ONLY the clinical trials listed below (if any) as the sole grounded "
            "information."
        )
    elif not paper_pool:
        grounding_note = (
            "NO papers were retrieved. Do not synthesize from training knowledge — state that the "
            "database does not contain evidence for this topic. Report only trials below (if any)."
        )
    else:
        note = f"{len(paper_pool)} papers retrieved. Cite a PMID only for claims stated in that paper's excerpt below."
        if ungrounded_terms:
            note += (
                f" IMPORTANT: the database has NO evidence for: {', '.join(ungrounded_terms)}. "
                "Say so explicitly and never attach a PMID to any claim about those terms."
            )
        note += (
            " Papers marked evidence_tier='landscape_mention' name a compound only in their full "
            "text (e.g. a drug-pipeline table), not their abstract — when citing such a paper for "
            "that compound, label the citation as a full-text/pipeline-table mention, not a primary study."
        )
        grounding_note = note

    # Typo suggestions (never substituted into the search). Surface as "did you mean?".
    if did_you_mean:
        hints = "; ".join(f"'{k}' → '{v}'" for k, v in did_you_mean.items())
        grounding_note += (
            f" POSSIBLE TYPOS (unrecognized query terms with a near match in the database): {hints}. "
            "If a suggestion looks right, tell the physician there was no exact match and ask whether "
            "they meant the suggested name, inviting them to re-query with it. Do NOT assume the "
            "suggestion is correct and do NOT search it yourself."
        )

    # Evidence tier — for each paper, flag focus terms (drug codes) that appear only in its
    # full text, not its abstract. A compound named only in a full-text pipeline/landscape
    # table means the paper is not a primary source for it; the synthesis model labels such
    # citations accordingly. Checks the paper's full concatenated text (all chunks), because
    # the retrieved representative chunk often is not the one holding the compound name.
    texts_by_pmid = rag_retriever.paper_texts_for_pmids(
        collection, [r["pmid"] for r in paper_pool]
    )
    # Only compounds that are landscape-only across the WHOLE corpus (absent from every
    # abstract, e.g. SPG302) can be reliably flagged from an abstract-vs-fulltext check.
    # A common gene like C9orf72 is discussed in many paper bodies without appearing in
    # their abstract — flagging those would wrongly demote primary studies, so restrict
    # the check to landscape_only_terms.
    fulltext_only_by_pmid: dict[str, list[str]] = {}
    for r in paper_pool:
        pmid = r["pmid"]
        texts = texts_by_pmid.get(pmid, {"abstract": "", "full": ""})
        fulltext_only_by_pmid[pmid] = [
            t for t in landscape_only_terms
            if rag_retriever.term_matches_text(texts["full"], t)
            and not rag_retriever.term_matches_text(texts["abstract"], t)
        ]

    _landscape = {pmid: terms for pmid, terms in fulltext_only_by_pmid.items() if terms}
    if _landscape:
        _logger.info("landscape-mention citations flagged", extra={"data": {"papers": _landscape}})

    return {
        "papers": [
            {
                "pmid": r["pmid"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "citation_count": r["citation_count"],
                "section": r["section"],
                "excerpt": _term_excerpt(r["document"], keyword_terms),
                "score": round(r["score"], 3),
                "fulltext_only_mentions": fulltext_only_by_pmid.get(r["pmid"], []),
                "evidence_tier": "landscape_mention" if fulltext_only_by_pmid.get(r["pmid"]) else "primary",
            }
            for r in paper_pool
        ],
        "query_entities": query_entities,
        "expanded_entities": expanded_entities,
        "ungrounded_terms": ungrounded_terms,
        "did_you_mean": did_you_mean,
        "trials": related_trials,
        "evidence_count": len(paper_pool),
        "kg_expansion_active": graph is not None,
        "grounding_note": grounding_note,
    }
