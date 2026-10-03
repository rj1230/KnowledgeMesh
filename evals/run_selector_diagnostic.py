import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals.selector_diagnostic import production_selector_diagnostic
from evals.selector_shadow import gold_ids, identity
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

PROBES = ROOT_DIR / "evals/datasets/inventory_retrieval_probes_70.json"

WANTED = {
    "inv_loop_001",
    "inv_harness_011",
    "inv_rag_008",
    "inv_rag_009",
    "inv_training_005",
}

data = json.loads(PROBES.read_text(encoding="utf-8"))

for probe in data:
    if probe["id"] not in WANTED:
        continue

    question = probe["question"]
    gold = gold_ids(probe)

    dense = search_enterprise_knowledge(
        question,
        limit=50,
    )

    reranked = rerank_documents(
        question,
        dense,
        top_n=50,
    )

    final, diag = production_selector_diagnostic(
        question,
        reranked,
        gold,
    )

    dense_ids = [identity(x) for x in dense]
    rerank_ids = [identity(x) for x in reranked]
    final_ids = [identity(x) for x in final]

    dense_gold = [
        (str(gid), dense_ids.index(gid) + 1)
        for gid in gold
        if gid in dense_ids
    ]

    rerank_gold = [
        (str(gid), rerank_ids.index(gid) + 1)
        for gid in gold
        if gid in rerank_ids
    ]

    final_gold = [
        (str(gid), final_ids.index(gid) + 1)
        for gid in gold
        if gid in final_ids
    ]

    print("\n" + "=" * 110)
    print(probe["id"])
    print("=" * 110)
    print(question)
    print("GOLD:", sorted(str(x) for x in gold))

    print("\nSTAGE SURVIVAL")
    print("Dense@50 :", dense_gold or "MISS")
    print("Rerank@50:", rerank_gold or "MISS")
    print(
        "Candidate:",
        "YES" if diag["gold_in_candidates"] else "NO",
    )
    print(
        "Selected before final sort:",
        "YES" if diag["gold_selected_before_final_sort"] else "NO",
    )
    print("Final@5:", final_gold or "MISS")

    if diag["gold_candidate_details"]:
        print("\nGOLD CANDIDATE")

        for item in diag["gold_candidate_details"]:
            print(
                f"  {item['id']}"
                f"  rerank={item['rerank_score']:.6f}"
                f"  dense={item['dense_score']:.6f}"
                f"  coverage={item['coverage']:.4f}"
                f"  recovery={item['recovery']}"
            )

    print("\nSELECTION TRACE")

    for step in diag["selection_steps"]:
        gold_rows = step["gold"]

        if not gold_rows:
            continue

        print(
            f"  step={step['step']}"
            f" chosen={step['chosen']}"
            f" chosen_score={step['chosen_score']:.6f}"
        )

        for row in gold_rows:
            print(
                f"    GOLD {row['id']}"
                f" rank={row['rank_among_remaining']}"
                f" sel={row['selection_score']:.6f}"
                f" rerank={row['rerank_score']:.6f}"
                f" dense={row['dense_score']:.6f}"
                f" cov={row['coverage']:.4f}"
                f" marginal={row['marginal_coverage']:.4f}"
                f" redundancy={row['redundancy']:.4f}"
                f" recovery={row['recovery']}"
            )

    print("\nSELECTED BEFORE FINAL SORT")
    print("  ", diag.get("pre_sort_selected", []))

    print("\nFINAL@5")
    print("  ", diag.get("final_selected", []))
