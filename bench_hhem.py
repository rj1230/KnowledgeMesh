import statistics
import time

from app.tools import entailment_tool as et

SENTENCE = (
    "Loop engineering is the discipline of designing loop specifications, "
    "external bounded reusable artifacts with a trigger, a goal, a "
    "verification step, a stopping rule, and a memory. "
)


def make_pairs(n: int = 8, repeats: int = 12):
    return [
        (
            (SENTENCE.replace("Loop", f"Variant {i} of loop")) * (repeats + i),
            f"Claim {i}: loop engineering designs loop specifications.",
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    et.warmup_hhem()
    pairs = make_pairs()
    times = []

    for _ in range(6):
        et.clear_hhem_cache()
        started = time.perf_counter()
        et.score_entailment_batch(pairs)
        times.append(time.perf_counter() - started)

    print(f"first call : {times[0]:.2f}s")
    print(f"median rest: {statistics.median(times[1:]):.2f}s")
    print(f"all        : {[round(t, 2) for t in times]}")
