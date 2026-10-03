import json
from pathlib import Path

path = Path("evals/results/inventory_probe_audit.json")
data = json.loads(path.read_text(encoding="utf-8"))

labels = {
    "inv_attention_001": (
        "A_EXACT",
        "Gold chunk directly states the Transformer architecture is based solely on attention and removes recurrence/convolution."
    ),
    "inv_attention_002": (
        "A_EXACT",
        "Gold chunk directly describes encoder and decoder self-attention position access."
    ),
    "inv_attention_003": (
        "A_EXACT",
        "Gold chunk contains the experiment table varying attention heads and key size."
    ),
    "inv_attention_004": (
        "A_EXACT",
        "Gold chunk contains the visualized attention/anaphora example."
    ),
    "inv_harness_001": (
        "C_WEAK_GOLD",
        "Abstract chunk is topically relevant but may not contain the detailed automation argument."
    ),
    "inv_harness_002": (
        "C_WEAK_GOLD",
        "Gold is a references chunk; unsuitable as a primary semantic retrieval target."
    ),
    "inv_harness_003": (
        "A_EXACT",
        "Gold chunk contains concrete research/search information for coding-agent techniques."
    ),
    "inv_harness_004": (
        "A_EXACT",
        "Gold chunk directly contains the recurring-coordination-patterns claim."
    ),
    "inv_llm_agents_001": (
        "A_EXACT",
        "Gold chunk contains the survey framing and definition of LLM-based agents."
    ),
    "inv_llm_agents_002": (
        "E_MULTI_CHUNK",
        "Gold chunk starts the Minecraft/embodied-agent discussion, but the requested planning details span surrounding content."
    ),
    "inv_llm_agents_003": (
        "A_EXACT",
        "Gold chunk directly discusses AutoGPT and its role."
    ),
    "inv_llm_agents_004": (
        "E_MULTI_CHUNK",
        "The question concerns task decomposition and the evidence spans the embodied-agent discussion."
    ),
    "inv_llm_001": (
        "A_EXACT",
        "Gold chunk directly states the motivation for pretraining and few-shot learning."
    ),
    "inv_llm_002": (
        "A_EXACT",
        "Gold chunk directly reports the 65% few-shot accuracy result."
    ),
    "inv_llm_003": (
        "E_MULTI_CHUNK",
        "The question asks for variation across a broad benchmark table rather than one isolated fact."
    ),
    "inv_llm_004": (
        "D_INVALID_GOLD",
        "Gold chunk is bibliography/reference material rather than substantive limitations or related-work discussion."
    ),
    "inv_rag_001": (
        "A_EXACT",
        "Gold chunk contains the RAG motivation and limitations of pretrained models."
    ),
    "inv_rag_002": (
        "A_EXACT",
        "Gold chunk directly explains the title-generation posterior behavior."
    ),
    "inv_rag_003": (
        "A_EXACT",
        "Gold chunk directly provides evidence about retrieval and generation during title generation."
    ),
    "inv_rag_004": (
        "A_EXACT",
        "Gold chunk explicitly introduces retrieval collapse."
    ),
    "inv_training_001": (
        "A_EXACT",
        "Gold chunk directly discusses why scaling model size alone does not solve instruction following."
    ),
    "inv_training_002": (
        "A_EXACT",
        "Gold chunk directly discusses generalization beyond supervised settings."
    ),
    "inv_training_003": (
        "A_EXACT",
        "Gold chunk directly describes the toxicity evaluation procedure."
    ),
    "inv_training_004": (
        "A_EXACT",
        "Gold chunk contains the exact human-evaluation figure caption comparing GPT-3 and InstructGPT."
    ),
    "inv_loop_001": (
        "A_EXACT",
        "Gold chunk explicitly lists trigger, goal, verification step, stopping rule, and memory."
    ),
    "inv_loop_002": (
        "A_EXACT",
        "Gold chunk directly explains Ralph's fresh-context and external-file state mechanism."
    ),
    "inv_loop_003": (
        "A_EXACT",
        "Gold chunk directly states that the loop automates typing rather than judgment."
    ),
    "inv_loop_004": (
        "E_MULTI_CHUNK",
        "Gold chunk introduces the five design dimensions and begins the Loop Library analysis; evidence spans adjacent material."
    ),
}

missing = sorted(set(item["id"] for item in data) - set(labels))
extra = sorted(set(labels) - set(item["id"] for item in data))

if missing:
    raise SystemExit(f"Missing labels: {missing}")

if extra:
    raise SystemExit(f"Unknown labels: {extra}")

for item in data:
    label, reason = labels[item["id"]]
    item["audit_label"] = label
    item["audit_reason"] = reason

output = Path("evals/results/inventory_probe_audit_labeled.json")
output.write_text(
    json.dumps(data, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

from collections import Counter

counts = Counter(labels[item["id"]][0] for item in data)

print("AUDIT CLASSIFICATION")
print("=" * 60)

for label in sorted(counts):
    print(f"{label:20s}: {counts[label]}")

print()
print(f"Total: {len(data)}")
print(f"Output: {output}")
