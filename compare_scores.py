import json
import sys

from app.tools import entailment_tool as et

FILLER = [
    "Vector databases index embeddings for approximate nearest-neighbour search.",
    "Rate limiting protects an API from bursts of requests.",
    "Containers package an application with its dependencies.",
    "Gradient descent updates parameters in the direction of lower loss.",
    "Caching stores the result of an expensive call for reuse.",
    "A load balancer spreads traffic across several servers.",
    "Tokenizers split text into subword units before a model sees it.",
    "Logging records events so failures can be diagnosed later.",
]

SUPPORT = (
    "Loop engineering is the discipline of designing loop specifications: "
    "external, bounded, reusable artifacts made of a trigger, a goal, a "
    "verification step, a stopping rule and a memory."
)


def long_premise(position: int) -> str:
    sentences = [FILLER[i % len(FILLER)] + f" (note {i})" for i in range(30)]
    sentences.insert(position, SUPPORT)
    return " ".join(sentences)


SHORT = SUPPORT + " It is positioned as a layer above prompt and context engineering."

PAIRS = [
    # (premise, claim)
    (SHORT, "Loop engineering designs loop specifications with a trigger and a goal."),
    (SHORT, "Loop engineering replaces prompt engineering entirely."),
    (SHORT, "A loop specification includes a verification step and a stopping rule."),
    (SHORT, "Loop engineering is used to train language models from scratch."),
    (
        long_premise(3),
        "Loop engineering designs loop specifications with a trigger and a goal.",
    ),
    (
        long_premise(15),
        "Loop engineering designs loop specifications with a trigger and a goal.",
    ),
    (long_premise(28), "A loop specification includes a memory and a stopping rule."),
    (long_premise(15), "Loop engineering requires a GPU cluster."),
]

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "save":
        et.warmup_hhem()
        et.clear_hhem_cache()
        scores = et.score_entailment_batch(PAIRS)
        with open(sys.argv[2], "w") as f:
            json.dump(scores, f)
        print("saved", sys.argv[2])

    elif mode == "diff":
        a = json.load(open(sys.argv[2]))
        b = json.load(open(sys.argv[3]))
        flips = 0
        for i, (x, y) in enumerate(zip(a, b)):
            flipped = (x >= 0.5) != (y >= 0.5)
            flips += flipped
            print(
                f"{i}: {x:.3f} -> {y:.3f}  delta {y - x:+.3f}  {'FLIP' if flipped else ''}"
            )
        print(f"verdict flips: {flips}")
