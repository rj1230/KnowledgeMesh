import logging
import os
import statistics
import time

from app.tools import entailment_tool as et


logging.basicConfig(
    level=os.getenv("BENCH_LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
)

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


def make_short_pairs(n: int = 8):
    return [
        (
            SENTENCE,
            f"Claim {i}: loop engineering designs loop specifications.",
        )
        for i in range(n)
    ]


def benchmark(label: str, pairs, runs: int = 6):
    times = []

    print(f"\n=== {label}: {len(pairs)} pairs, batch_size={et.HHEM_BATCH_SIZE} ===")

    for _ in range(runs):
        et.clear_hhem_cache()

        started = time.perf_counter()
        scores = et.score_entailment_batch(pairs)
        elapsed = time.perf_counter() - started

        if len(scores) != len(pairs):
            raise RuntimeError(f"Expected {len(pairs)} scores, received {len(scores)}")

        times.append(elapsed)

    first = times[0]
    rest = times[1:]
    median = statistics.median(rest)

    print(f"first call : {first:.2f}s")
    print(f"median rest: {median:.2f}s")
    print(f"all        : {[round(t, 2) for t in times]}")
    print(f"throughput : {len(pairs) / median:.2f} pairs/s")


if __name__ == "__main__":
    et.warmup_hhem()

    benchmark("short realistic pairs", make_short_pairs())
    benchmark("long stress pairs", make_pairs())
