"""
Retrieval quality, measured.

Every other suite checks that retrieval *runs*. None of them checks whether it finds the
right passage, and "the answers looked good" is not evidence: almost every question this
project has been tested with was answered from a dataframe by a tool, not from retrieved
text. The RAG half was the least exercised part of the system and the only one with no
number attached to it.

This builds a corpus whose right answer is known for each question, then reports recall@1,
recall@5 and MRR across the retrieval configurations the settings actually offer. It needs
the real embedder and reranker, so it is slow and excluded from discovery — run it the way
e2e.py is run. It makes no network calls and costs nothing.

    docker compose exec backend python tests/retrieval_eval.py

The corpus is built to separate the two halves of hybrid search rather than to flatter it:
lexical questions whose answer shares rare tokens with the passage, semantic questions
phrased with none of the passage's words, a cross-language pair, and near-duplicate
distractors that punish a retriever which matches on topic alone.
"""

import asyncio
import sys
import time
from collections import defaultdict

from app.core.config import get_settings
from app.core.vectorstore import VectorStore
from app.services.retrieval import HybridRetriever

settings = get_settings()
SESSION = "eval"

# (id, text). The id is what a question names as its correct answer.
CORPUS = [
    # --- rare tokens: lexical retrieval should carry these -----------------------
    ("sla-credit",
     "Service credits are governed by clause 7.4. A credit of up to 15% of the monthly "
     "fee may be authorised by the Regional Manager. Credits above 15% require Finance "
     "approval and a signed variance form VF-2231."),
    ("escalation",
     "A ticket is raised at Tier 1. If unresolved within 4 working hours it escalates to "
     "Tier 2, which has 2 working days. Anything still open goes to the Regional Manager."),
    ("penalty",
     "Late delivery beyond 5 working days incurs a penalty of 2% of order value per "
     "additional week, capped at 10% in total."),
    ("part-number",
     "Replacement compressors are stocked under part number CMP-4417-B. The superseded "
     "part CMP-4410-A is no longer available and must not be ordered."),
    ("vpn-config",
     "Remote engineers connect through the WireGuard endpoint vpn-riyadh-03.internal on "
     "UDP port 51820. The legacy OpenVPN concentrator was decommissioned in March."),

    # --- paraphrase only: semantic retrieval should carry these ------------------
    ("onboarding",
     "New joiners spend their first week shadowing a senior colleague before being given "
     "their own queue. Nobody handles customer tickets alone until that period ends."),
    ("burnout",
     "Staff who work more than two consecutive weekends are automatically flagged to their "
     "line manager, and the rota is adjusted before fatigue affects service quality."),
    ("procurement-delay",
     "Orders placed after the Thursday cut-off are not picked until the following Monday, "
     "which adds two days to every request made late in the week."),
    ("data-retention",
     "Customer records are kept for seven years after the final interaction, then removed "
     "in the quarterly purge. Anything older cannot be recovered once that job has run."),

    # --- cross-language: an English question must find Arabic text ---------------
    ("arabic-permit",
     "تصريح العمل الخاص يُمنح للأفراد الذين يمارسون عملاً شخصياً مستقلاً، ولا يسمح لهم "
     "بتوظيف عمال آخرين، ورسومه منخفضة ومدته سنة واحدة قابلة للتجديد."),

    # --- multi-hop: the answer needs two passages -------------------------------
    ("shift-pattern",
     "Engineers work four ten-hour days. The fifth day is covered by the on-call rota, "
     "which rotates weekly and is published a month in advance."),
    ("on-call-pay",
     "On-call weeks attract a 20% uplift on base pay for the days actually covered, "
     "claimed through the monthly timesheet rather than payroll."),

    # --- numeric passages, where a wrong hit is a wrong figure -------------------
    ("threshold-warning",
     "An alert fires when queue depth exceeds 250 items for more than 10 minutes. "
     "Below that the system self-drains and no action is required."),
    ("threshold-critical",
     "A critical page is raised at 900 items or a 45-minute sustained breach, whichever "
     "comes first, and goes directly to the duty engineer."),

    # --- negation, which embeddings famously blur --------------------------------
    ("refund-allowed",
     "Refunds are issued for goods returned within 30 days in original packaging."),
    ("refund-refused",
     "Refunds are not issued for bespoke or made-to-order items under any circumstances, "
     "regardless of condition or how recently they were delivered."),

    # --- near-duplicate distractors: same topic, wrong answer --------------------
    ("distractor-credit-policy",
     "Service credit requests are logged in the ticketing system for audit purposes. The "
     "log records who raised the request and when, but not the amount approved."),
    ("distractor-escalation-training",
     "All Tier 1 staff complete escalation training during induction. The training covers "
     "when to escalate but does not itself define the time limits."),
    ("distractor-penalty-history",
     "Penalties were last revised in 2023. The previous schedule used a flat fee rather "
     "than a percentage of order value."),
    ("distractor-vpn-history",
     "The network team has run remote access since 2019. Several technologies have been "
     "trialled over that period before the current arrangement settled."),
    ("distractor-retention-policy-owner",
     "The data retention policy is owned by the Compliance function and reviewed annually "
     "by the steering group."),
]

# (question, id it must find, why this question is here)
QUESTIONS = [
    ("What is the variance form number for service credits above 15%?", "sla-credit", "rare token"),
    ("Which part number replaces CMP-4410-A?", "part-number", "rare token"),
    ("What port does the WireGuard endpoint use?", "vpn-config", "rare token"),
    ("How long does Tier 2 have before a ticket goes to the Regional Manager?", "escalation", "rare token"),
    ("What is the cap on late delivery penalties?", "penalty", "rare token"),

    ("Can a new employee take calls on their own straight away?", "onboarding", "paraphrase"),
    ("What happens to someone rostered on too many weekends in a row?", "burnout", "paraphrase"),
    ("Why does ordering on a Friday take longer?", "procurement-delay", "paraphrase"),
    ("How long before old customer information is destroyed?", "data-retention", "paraphrase"),
    ("Is there anything stopping staff from becoming exhausted?", "burnout", "paraphrase"),
    ("What is the last point at which a record can still be retrieved?", "data-retention", "paraphrase"),

    ("What are the rules for an individual work permit?", "arabic-permit", "cross-language"),
    ("Can someone with a personal work permit hire employees?", "arabic-permit", "cross-language"),

    ("Who can approve a service credit, and up to what percentage?", "sla-credit", "distractor nearby"),
    ("What are the escalation time limits?", "escalation", "distractor nearby"),
    ("How is the late delivery penalty calculated?", "penalty", "distractor nearby"),
    ("How do remote engineers connect to the network today?", "vpn-config", "distractor nearby"),

    ("How much extra is paid for covering an on-call week?", "on-call-pay", "multi-hop"),
    ("How is the fifth working day covered?", "shift-pattern", "multi-hop"),
    ("At what queue depth does a critical page go out?", "threshold-critical", "numeric"),
    ("What queue depth only triggers a warning?", "threshold-warning", "numeric"),
    ("When can a customer not get a refund?", "refund-refused", "negation"),
    ("What are the conditions for a refund to be granted?", "refund-allowed", "negation"),
]

CONFIGS = [
    ("BM25 only",            {"BM25_WEIGHT": 1.0, "SEMANTIC_WEIGHT": 0.0, "USE_RERANKING": False}),
    ("semantic only",        {"BM25_WEIGHT": 0.0, "SEMANTIC_WEIGHT": 1.0, "USE_RERANKING": False}),
    ("hybrid, no rerank",    {"BM25_WEIGHT": 0.3, "SEMANTIC_WEIGHT": 0.7, "USE_RERANKING": False}),
    ("hybrid + rerank",      {"BM25_WEIGHT": 0.3, "SEMANTIC_WEIGHT": 0.7, "USE_RERANKING": True}),
    ("lexical-heavy hybrid", {"BM25_WEIGHT": 0.7, "SEMANTIC_WEIGHT": 0.3, "USE_RERANKING": True}),
]


def rank_of(target, hits):
    """1-based rank of the wanted chunk, or None if it never appeared."""
    for position, hit in enumerate(hits, start=1):
        if hit.get("chunk_id") == target:
            return position
    return None


async def main():
    store = VectorStore()
    # Isolated from the running instance's index in both directions: without this the
    # persisted corpus is restored underneath the fixtures, the fixtures are appended to
    # it, and a second run measures a corpus containing two copies of itself. The first
    # version of this harness drifted 6 points between runs for exactly that reason.
    store.persist = False
    await store.initialize()
    store.documents, store.index = [], __import__("faiss").IndexFlatIP(store.dim)
    await store.add_documents([
        {"content": text, "source": "eval.txt", "session_id": SESSION,
         "metadata": {"chunk_id": cid}, "chunk_id": cid}
        for cid, text in CORPUS
    ])
    retriever = HybridRetriever(store)
    await retriever._initialize_reranker()
    print(f"corpus: {len(CORPUS)} chunks   questions: {len(QUESTIONS)}\n")

    settings.RETRIEVAL_TOP_K = 5
    baseline = (settings.BM25_WEIGHT, settings.SEMANTIC_WEIGHT, settings.USE_RERANKING)
    rows, per_kind = [], defaultdict(lambda: defaultdict(list))

    for name, overrides in CONFIGS:
        settings.BM25_WEIGHT = overrides["BM25_WEIGHT"]
        settings.SEMANTIC_WEIGHT = overrides["SEMANTIC_WEIGHT"]
        settings.USE_RERANKING = overrides["USE_RERANKING"]

        ranks, started = [], time.time()
        for question, target, kind in QUESTIONS:
            hits = await retriever.retrieve(question, session_ids={SESSION})
            rank = rank_of(target, hits)
            ranks.append(rank)
            per_kind[name][kind].append(rank)
        elapsed = time.time() - started

        found = [r for r in ranks if r]
        rows.append((
            name,
            sum(1 for r in ranks if r == 1) / len(ranks),
            len(found) / len(ranks),
            sum(1 / r for r in found) / len(ranks),
            elapsed / len(QUESTIONS),
        ))

    settings.BM25_WEIGHT, settings.SEMANTIC_WEIGHT, settings.USE_RERANKING = baseline

    print(f"{'configuration':<24}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}{'s/query':>9}")
    print("-" * 61)
    for name, r1, r5, mrr, secs in rows:
        print(f"{name:<24}{r1:>9.0%}{r5:>10.0%}{mrr:>8.2f}{secs:>9.2f}")

    print(f"\n{'by question type':<24}" + "".join(f"{k:>14}" for k in
          ("rare token", "paraphrase", "cross-language", "distractor nearby")))
    print("-" * 80)
    for name, _ in CONFIGS:
        cells = []
        for kind in ("rare token", "paraphrase", "cross-language", "distractor nearby"):
            rs = per_kind[name][kind]
            cells.append(f"{sum(1 for r in rs if r == 1) / len(rs):>13.0%}")
        print(f"{name:<24}" + "".join(cells))
    print("\nCells are recall@1 — the wanted passage ranked first.")

    # The reranker filters anything below RERANK_THRESHOLD, so a correct passage ranked
    # second can be discarded rather than demoted. That trades recall for precision, and
    # the trade is invisible without measuring it.
    settings.BM25_WEIGHT, settings.SEMANTIC_WEIGHT = 0.3, 0.7
    settings.USE_RERANKING = True
    print(f"\n{'RERANK_THRESHOLD':<20}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}"
          f"{'avg kept':>10}")
    print("-" * 58)
    for threshold in (0.0, 0.25, 0.35, 0.45, 0.60):
        settings.RERANK_THRESHOLD = threshold
        ranks, kept = [], []
        for question, target, _ in QUESTIONS:
            hits = await retriever.retrieve(question, session_ids={SESSION})
            ranks.append(rank_of(target, hits))
            kept.append(len(hits))
        found = [r for r in ranks if r]
        marker = "  <- shipped" if threshold == 0.35 else ""
        print(f"{threshold:<20.2f}{sum(1 for r in ranks if r == 1) / len(ranks):>9.0%}"
              f"{len(found) / len(ranks):>10.0%}"
              f"{sum(1 / r for r in found) / len(ranks):>8.2f}"
              f"{sum(kept) / len(kept):>10.1f}{marker}")
    settings.RERANK_THRESHOLD = 0.35

    shipped = next(r for r in rows if r[0] == "hybrid + rerank")
    print(f"\nShipped configuration: recall@1 {shipped[1]:.0%}, recall@5 {shipped[2]:.0%}, "
          f"MRR {shipped[3]:.2f}")
    failures = []
    settings.USE_RERANKING = True
    for question, target, kind in QUESTIONS:
        hits = await retriever.retrieve(question, session_ids={SESSION})
        if rank_of(target, hits) != 1:
            got = hits[0].get("chunk_id") if hits else "nothing"
            failures.append(f"  {kind:<18} {question}\n{'':22}wanted {target}, got {got}")
    if failures:
        print("\nQuestions the shipped configuration does not rank first:")
        print("\n".join(failures))
    settings.BM25_WEIGHT, settings.SEMANTIC_WEIGHT, settings.USE_RERANKING = baseline
    return 0


sys.exit(asyncio.run(main()))
