#!/usr/bin/env python3
"""Deterministic instance sampler for the SWE-bench Verified end-to-end eval (Phase 5 / E7).

Reproducibility contract
------------------------
* Dataset: ``princeton-nlp/SWE-bench_Verified`` (the 500-instance human-curated subset).
* Procedure: load all instance ids, **sort them lexicographically** (so the pool is
  order-independent of however ``datasets`` happens to yield rows), then draw a sample
  with ``random.Random(SEED).sample(sorted_ids, n)``.
* Seed: 1729 (fixed). Same seed + same ``n`` => byte-identical sample, on any machine.

The sample is the *only* place randomness enters Phase 5; everything downstream is keyed
by ``instance_id`` so a partial run can resume without reshuffling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SEED = 1729
DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"


def load_verified() -> list[dict[str, Any]]:
    """Return all SWE-bench Verified rows as plain dicts (one network/cache hit)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT)
    return [dict(row) for row in ds]


def sample_instance_ids(
    all_rows: list[dict[str, Any]], n: int, seed: int = SEED
) -> list[str]:
    """Deterministically draw ``n`` instance ids from the full pool.

    Sorting first makes the draw independent of dataset row order; the seeded
    ``random.Random`` makes it reproducible.
    """
    pool = sorted(row["instance_id"] for row in all_rows)
    if n > len(pool):
        raise ValueError(f"requested n={n} > pool size {len(pool)}")
    return random.Random(seed).sample(pool, n)


def build_sample(n: int, seed: int = SEED) -> dict[str, Any]:
    rows = load_verified()
    ids = set(sample_instance_ids(rows, n, seed))
    by_id = {r["instance_id"]: r for r in rows}
    chosen = [by_id[i] for i in sorted(ids)]
    return {
        "dataset": DATASET,
        "split": SPLIT,
        "seed": seed,
        "n_requested": n,
        "pool_size": len(rows),
        "instance_ids": sorted(ids),
        "instances": chosen,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=50, help="sample size (default 50)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/paper/results/swe_e2e/sample.json"),
        help="where to write the resolved sample manifest",
    )
    args = ap.parse_args()
    sample = build_sample(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Manifest (ids only) is tiny and committable; the full instances blob stays local.
    manifest = {k: v for k, v in sample.items() if k != "instances"}
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    full = args.out.with_name(args.out.stem + "_full.json")
    full.write_text(json.dumps(sample, indent=2) + "\n")
    print(f"sample: n={args.n} seed={args.seed} pool={sample['pool_size']}")
    print(f"manifest -> {args.out}")
    print(f"full     -> {full}")
    for i in sample["instance_ids"]:
        print("  ", i)


if __name__ == "__main__":
    main()
