"""Self-calibrating token counts — correct the offline heuristic against the provider's own
billed usage, which distil (being a proxy) sees on every response.

The offline heuristic (`tokenizer.HeuristicTokenizer`) is directionally accurate but not
billing-grade — it can be ~15-20% off the real BPE, more on code. But distil holds a unique
pairing: for every live request it has both the text *and* the API's `usage.input_tokens`. This
module accumulates that pairing and learns the systematic correction, so reported token counts
converge to the real tokenizer for *your* model + content mix — with no per-string network call.

Two properties make this safe to put on the headline (public-facing):
  - **Scale-invariant on the percentage.** Savings % = 1 − distil/baseline is unchanged by a
    uniform factor; calibration only corrects the *absolute* counts (the "1.02B tokens" figure),
    never the "50% smaller" claim.
  - **Identity until proven.** With fewer than ``MIN_SAMPLES`` observations the factor is exactly
    1.0, so an uncalibrated install reports precisely what it does today — no regression.

**Content-free.** Only integer token counts (heuristic estimate, billed total) and the model id
are stored — never a prompt, a line, or any text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MIN_SAMPLES = 20  # below this, don't trust the factor — report the raw heuristic (identity)
_RESERVOIR = 500  # bounded per-model ratio history for a robust median + CI
# Plausibility bounds for a *tokenizer* correction. Real heuristics are off by ~0.7-1.7x; a
# factor outside this window means the data is unreliable (mis-recorded usage, prompt-cache
# accounting error, test noise), not a genuine tokenizer difference — fall back to identity so
# a bad signal can never poison a public-facing headline.
_MIN_FACTOR = 0.5
_MAX_FACTOR = 3.0
_STORE_VERSION = 1


def _path() -> Path:
    return Path(os.environ.get("DISTIL_HOME", str(Path.home() / ".distil"))) / "calibration.json"


def _load(path: Path | None = None) -> dict:
    p = path or _path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("v") == _STORE_VERSION:
            return data
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {"v": _STORE_VERSION, "models": {}}


def record(model: str, est_tokens: int, billed_tokens: int, *, path: Path | None = None) -> None:
    """Record one (heuristic estimate, billed) pairing for *model*. Content-free, fail-open —
    the calibration signal must never slow or break a request. Skips non-positive counts."""
    if est_tokens <= 0 or billed_tokens <= 0:
        return
    try:
        p = path or _path()
        data = _load(p)
        m = data["models"].setdefault(
            model or "unknown", {"est_sum": 0, "billed_sum": 0, "n": 0, "ratios": []}
        )
        m["est_sum"] += int(est_tokens)
        m["billed_sum"] += int(billed_tokens)
        m["n"] += 1
        ratios = m["ratios"]
        ratios.append(round(billed_tokens / est_tokens, 4))
        if len(ratios) > _RESERVOIR:  # keep the most recent — the current tokenizer/content mix
            del ratios[: len(ratios) - _RESERVOIR]
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        tmp.replace(p)  # atomic — a torn write can't corrupt the store
    except Exception:  # noqa: BLE001 — calibration bookkeeping never breaks the request path
        pass


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 1.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def factor(model: str | None = None, *, path: Path | None = None) -> tuple[float, int]:
    """Return ``(calibration_factor, n_samples)``. The factor multiplies a heuristic token count
    to estimate the billed count. Robust (aggregate ratio, cross-checked by the median). Returns
    ``(1.0, n)`` until at least ``MIN_SAMPLES`` observations — identity, so no early skew.

    With no *model* (or an unseen one) the pooled all-models aggregate is used."""
    data = _load(path)
    models = data.get("models", {})
    if model and model in models:
        m = models[model]
        est, billed, n, ratios = m["est_sum"], m["billed_sum"], m["n"], m.get("ratios", [])
    else:  # pool across models
        est = sum(v["est_sum"] for v in models.values())
        billed = sum(v["billed_sum"] for v in models.values())
        n = sum(v["n"] for v in models.values())
        ratios = [r for v in models.values() for r in v.get("ratios", [])]
    if n < MIN_SAMPLES or est <= 0:
        return 1.0, n
    aggregate = billed / est
    # Guard against a pathological aggregate (a few giant requests skewing the sum) by blending
    # with the median of per-request ratios; both agree in the common case.
    med = _median(ratios) if ratios else aggregate
    f = round((aggregate + med) / 2.0, 4)
    # Sanity gate: a plausible tokenizer correction lives in [_MIN_FACTOR, _MAX_FACTOR]. Anything
    # outside is a data problem, not a tokenizer difference — return identity so it can never
    # poison the headline.
    if not (_MIN_FACTOR <= f <= _MAX_FACTOR):
        return 1.0, n
    return f, n


def relative_ci(model: str | None = None, *, path: Path | None = None) -> float | None:
    """Relative half-width of the calibration (IQR/median over the ratio reservoir), or None when
    uncalibrated. A small number means the correction is precise for your traffic."""
    data = _load(path)
    models = data.get("models", {})
    if model and model in models:
        ratios = models[model].get("ratios", [])
    else:
        ratios = [r for v in models.values() for r in v.get("ratios", [])]
    if len(ratios) < MIN_SAMPLES:
        return None
    s = sorted(ratios)
    q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
    med = _median(s)
    return round((q3 - q1) / 2.0 / med, 3) if med else None


def calibrate(count: int, model: str | None = None, *, path: Path | None = None) -> int:
    """Apply the calibration factor to a heuristic token *count*."""
    f, _n = factor(model, path=path)
    return int(round(count * f))


def status(model: str | None = None, *, path: Path | None = None) -> dict:
    """A small report dict for the leaderboard/dashboard: factor, samples, CI, calibrated flag."""
    f, n = factor(model, path=path)
    # "calibrated" means a *real* correction is being applied. A factor of exactly 1.0 — whether
    # genuinely (heuristic already matches) or because the sanity gate rejected bad data — means
    # the displayed number IS the raw heuristic, so we don't claim calibration.
    return {
        "factor": f,
        "samples": n,
        "relative_ci": relative_ci(model, path=path),
        "calibrated": n >= MIN_SAMPLES and f != 1.0,
    }


if __name__ == "__main__":  # self-check — convergence, identity, content-free, no framework
    import tempfile

    p = Path(tempfile.mkdtemp()) / "cal.json"
    # heuristic under-counts by a true factor of 1.2 (billed = 1.2 * est), with a little noise
    for i in range(50):
        est = 1000 + i
        record("claude-x", est, int(est * 1.2) + (i % 3 - 1), path=p)
    f, n = factor("claude-x", path=p)
    assert n == 50 and 1.15 <= f <= 1.25, (f, n)
    assert calibrate(1000, "claude-x", path=p) == round(1000 * f)
    # identity below MIN_SAMPLES
    p2 = Path(tempfile.mkdtemp()) / "cal2.json"
    record("m", 100, 130, path=p2)
    assert factor("m", path=p2) == (1.0, 1), "must be identity until MIN_SAMPLES"
    # content-free: the store holds only integers + the model id
    blob = p.read_text()
    assert "claude-x" in blob and all(c not in blob for c in ("prompt", "text", "message"))
    print(f"calibration self-check ok: factor={f} n={n} ci={relative_ci('claude-x', path=p)}")
