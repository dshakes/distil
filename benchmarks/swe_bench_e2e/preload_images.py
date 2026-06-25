#!/usr/bin/env python3
"""Side-load SWE-bench instance images into the local Docker daemon (Phase 5 / E7).

Why this exists
---------------
On this machine the Docker daemon's *registry-proxy pull path*
(``http.docker.internal:3128``) is wedged: ``docker pull`` of any uncached image hangs
indefinitely, while the daemon's local **run** path is healthy (verified: a local amd64
image runs under emulation, ``EMU_OK x86_64``). A Docker Desktop restart would clear the
pull path but would also kill the user's long-running ``kage/*`` dev containers, so it is
off-limits for an unattended run.

The official SWE-bench harness only needs the per-instance images to be **present
locally** (with ``--namespace swebench`` it logs "Found N existing instance images. Will
reuse them." and skips the pull). We therefore fetch each prebuilt image *host-side* with
``skopeo`` (the host network reaches Docker Hub fine — ~20 s/image) into a docker-archive
tar, then ``docker load`` it (no registry access). This is a pure transport workaround:
the images, the harness, and the scoring are 100% the official SWE-bench pipeline.

Idempotent + resumable: images already in the daemon are skipped; tars are streamed to a
temp path and removed after load.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "swebench"
ARCH = "x86_64"


def image_ref(instance_id: str) -> str:
    """SWE-bench instance image reference for ``--namespace swebench`` (matches the harness).

    The harness builds this key as ``{ns}/sweb.eval.{arch}.{id}:latest`` with the instance
    id lowercased and ``__`` replaced by ``_1776_`` (SWE-bench's fixed sentinel for the
    double underscore, which is illegal in a Docker tag). Verified to match
    ``make_test_spec(...).instance_image_key`` exactly for the sampled instances.
    """
    norm = instance_id.replace("__", "_1776_").lower()
    return f"{NAMESPACE}/sweb.eval.{ARCH}.{norm}:latest"


def have_image(ref: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", ref], capture_output=True, text=True
    )
    return r.returncode == 0


def fetch_and_load(instance_id: str, tar_dir: Path, timeout: float = 600.0) -> dict:
    ref = image_ref(instance_id)
    if have_image(ref):
        return {
            "instance_id": instance_id,
            "ref": ref,
            "status": "cached",
            "seconds": 0.0,
        }
    tar = tar_dir / f"{instance_id}.tar"
    t0 = time.time()
    cp = subprocess.run(
        [
            "skopeo",
            "copy",
            "--override-arch",
            "amd64",
            "--override-os",
            "linux",
            f"docker://{ref}",
            f"docker-archive:{tar}:{ref}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if cp.returncode != 0:
        tar.unlink(missing_ok=True)
        return {
            "instance_id": instance_id,
            "ref": ref,
            "status": "skopeo_failed",
            "error": cp.stderr[-400:],
            "seconds": round(time.time() - t0, 1),
        }
    ld = subprocess.run(
        ["docker", "load", "-i", str(tar)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tar.unlink(missing_ok=True)
    status = "loaded" if ld.returncode == 0 else "load_failed"
    return {
        "instance_id": instance_id,
        "ref": ref,
        "status": status,
        "error": "" if ld.returncode == 0 else ld.stderr[-400:],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample", type=Path, default=ROOT / "docs/paper/results/swe_e2e/sample.json"
    )
    ap.add_argument("--tar-dir", type=Path, default=ROOT / ".e7_cache/tars")
    ap.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/paper/results/swe_e2e/preload_report.json",
    )
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    ids = json.loads(args.sample.read_text())["instance_ids"]
    if args.only:
        want = set(args.only.split(","))
        ids = [i for i in ids if i in want]
    args.tar_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n, iid in enumerate(ids, 1):
        print(f"[{n}/{len(ids)}] {iid} ...", flush=True)
        try:
            res = fetch_and_load(iid, args.tar_dir)
        except subprocess.TimeoutExpired:
            res = {"instance_id": iid, "ref": image_ref(iid), "status": "timeout"}
        results.append(res)
        print(f"    {res['status']} ({res.get('seconds', 0)}s)", flush=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")

    ok = sum(1 for r in results if r["status"] in ("loaded", "cached"))
    print(f"\npreloaded {ok}/{len(ids)} images; report -> {args.report}")
    if ok < len(ids):
        print(
            "FAILURES:",
            [
                r["instance_id"]
                for r in results
                if r["status"] not in ("loaded", "cached")
            ],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
