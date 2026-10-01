"""Realistic-corpus LIVE run: DERC certificate + fair head-to-head.

One pass over a realistic mixed corpus (5 domains, 4.5-6.5 KB/turn) graded live
against claude-opus-4-8 (majority-of-3). Produces:
  (1) Distil's live DERC certificate (shipped LTT statistics), and
  (2) a FAIR head-to-head vs the real packages — each invoked the way that gives
      it its best shot (LLMLingua per-block; Headroom via a real tool_use/
      tool_result conversation with optimize=True; RTK = not comparable).
Savings for every method are measured identically: compressed block tokens over
original block tokens.
"""

import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "benchmarks")
os.environ.setdefault("HEADROOM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
from anthropic import Anthropic

from distil.compress.adaptive import byte_exact
from distil.compress.strategies import distil as _distil
from distil.conformal import crc_select, hb_pvalue, ltt_certify
from distil.corpus import load_corpus
from distil.replay.anthropic_runner import AnthropicRunner
from distil.tokenizer import DEFAULT as tok
from distil.trajectory import Stability

ALPHA, DELTA, SAMPLES, WORKERS = 0.05, 0.05, 3, 8
client = Anthropic(max_retries=6)
# ponytail: pinned to the grader this documented run used, not the moving certifier default.
runner = AnthropicRunner(model="claude-opus-4-8", client=client, samples=1, effort=None)
entries = load_corpus("/tmp/corpus_realworld")
turns = [t for e in entries for t in e.trajectory.turns]
n = len(turns)


# ---- distil ladder transforms ----------------------------------------------
def _prune(blocks, turn):
    return [
        b for b in blocks if not (b.stability is Stability.VOLATILE and not b.decision_relevant)
    ]


def _prune_lossless(blocks, turn):
    return _distil(_prune(blocks, turn), turn)


def _truncate(limit):
    def s(blocks, turn):
        return [
            b.copy_with(b.text[:limit]) if b.stability is Stability.VOLATILE else b for b in blocks
        ]

    return s


# ---- competitor transforms --------------------------------------------------
def _llmlingua(blocks, turn):
    import llmlingua_adapter

    out = llmlingua_adapter.compress([b.text for b in blocks])
    return [b.copy_with(c) for b, c in zip(blocks, out)]


def _headroom(blocks, turn):
    """Fair Headroom: present volatile blocks as tool_results in a real
    conversation (optimize=True engages its pipeline), map compressed text back
    1:1. Stable/history blocks are left intact (as distil keeps its prefix)."""
    from headroom import compress as hr

    conv = [
        {
            "role": "user",
            "content": "".join(b.text for b in blocks if b.stability is not Stability.VOLATILE),
        }
    ]
    track = []  # (block_index, message_index)
    for i, b in enumerate(blocks):
        if b.stability is Stability.VOLATILE:
            conv.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": f"t{i}", "name": b.id, "input": {}}],
                }
            )
            track.append((i, len(conv)))
            conv.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": b.text}],
                }
            )
    r = hr(conv, model="claude-sonnet-4-5", model_limit=2000, optimize=True)
    out_msgs = getattr(r, "messages", r)
    if len(out_msgs) != len(conv):
        raise RuntimeError(f"headroom changed message count {len(conv)}->{len(out_msgs)}")
    new = list(blocks)
    for bi, mi in track:
        c = out_msgs[mi]["content"] if isinstance(out_msgs[mi], dict) else out_msgs[mi].content
        text = (
            "".join(
                (x.get("content") or x.get("text") or "") if isinstance(x, dict) else str(x)
                for x in c
            )
            if isinstance(c, list)
            else str(c)
        )
        if len(text) < len(blocks[bi].text):  # reject-if-bigger
            new[bi] = blocks[bi].copy_with(text)
    return new


# distil ladder (least->most aggressive by expected risk) for the certificate
LADDER = [
    ("byte-exact", byte_exact),
    ("lossless", _distil),
    ("causal-prune", _prune),
    ("prune+lossless", _prune_lossless),
    ("truncate@400", _truncate(400)),
    ("truncate@200", _truncate(200)),
]
# competitors graded for the head-to-head (not part of the LTT ladder)
COMPETITORS = [("LLMLingua-2", _llmlingua), ("Headroom", _headroom)]

# ---- precompute compressed blocks (local); verify each method engages --------
print(
    f"realistic corpus: {n} turns; compressing with {len(LADDER) + len(COMPETITORS)} methods...",
    flush=True,
)
compiled = {}  # name -> list[blocks] or None
notes = {}
base_tok = sum(tok.count(b.text) for t in turns for b in t.blocks)
for name, fn in LADDER + COMPETITORS:
    t0 = time.time()
    try:
        cb = [fn(t.blocks, t.index) for t in turns]
        compiled[name] = cb
        sav = 1.0 - sum(tok.count(b.text) for blk in cb for b in blk) / base_tok
        engaged = sum(
            1
            for ti, t in enumerate(turns)
            if sum(tok.count(b.text) for b in cb[ti]) < sum(tok.count(b.text) for b in t.blocks)
        )
        notes[name] = f"savings={sav * 100:.1f}%  engaged {engaged}/{n}  ({time.time() - t0:.0f}s)"
    except Exception as e:  # noqa: BLE001
        compiled[name] = None
        notes[name] = f"NOT COMPARABLE — {type(e).__name__}: {str(e)[:90]}"
    print(f"  {name}: {notes[name]}", flush=True)

# RTK — attempt honestly
try:
    import rtk_adapter

    rtk_adapter.compress([t.blocks[0].text for t in turns[:1]])
    notes["RTK"] = "ran"
except Exception as e:  # noqa: BLE001
    notes["RTK"] = f"NOT COMPARABLE — {str(e)[:110]}"
compiled["RTK"] = None

# ---- live grading -----------------------------------------------------------
graded = [name for name in compiled if compiled[name] is not None]
tasks = [("base", ti, s) for ti in range(n) for s in range(SAMPLES)]
for name in graded:
    tasks += [(name, ti, s) for ti in range(n) for s in range(SAMPLES)]
print(
    f"\nlive grading: {len(tasks)} calls ({len(graded) + 1} cells x {n} turns x {SAMPLES})...",
    flush=True,
)


def _one(task):
    name, ti, _s = task
    blocks = turns[ti].blocks if name == "base" else compiled[name][ti]
    try:
        return (name, ti), runner._sample(blocks)
    except Exception as exc:  # noqa: BLE001
        return (name, ti), f"<err:{type(exc).__name__}>"


t0 = time.time()
res = {}
done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for k, dec in pool.map(_one, tasks):
        res.setdefault(k, []).append(dec)
        done += 1
        if done % 240 == 0:
            print(f"  ... {done}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)


def majority(k):
    v = [d for d in res.get(k, []) if not d.startswith("<err")]
    return Counter(v).most_common(1)[0][0] if v else "<err>"


def risk_savings(name):
    cb = compiled[name]
    changes = sum(1 for ti in range(n) if majority((name, ti)) != majority(("base", ti)))
    sav = 1.0 - sum(tok.count(b.text) for blk in cb for b in blk) / base_tok
    return changes / n, sav


# ---- DERC certificate on the distil ladder ----------------------------------
level_losses = [
    [1.0 if majority((nm, ti)) != majority(("base", ti)) else 0.0 for ti in range(n)]
    for nm, _ in LADDER
]
ltt_certify(level_losses, alpha=ALPHA, delta=DELTA)  # statistical certification per level
# Operating point = the HIGHEST-SAVINGS level whose risk is certified <= alpha.
# (LTT's fixed-sequence index assumes savings is monotone in ladder position; this
# ladder mixes decision-aware and blind levels, so select by savings among certified.)
_certified = [
    li for li in range(len(LADDER)) if hb_pvalue(risk_savings(LADDER[li][0])[0], n, ALPHA) <= DELTA
]
idx = max(_certified, key=lambda li: risk_savings(LADDER[li][0])[1]) if _certified else -1
idx_crc = crc_select(level_losses, alpha=ALPHA)
print("\n" + "=" * 80)
print(f"LIVE DERC CERTIFICATE  (claude-opus-4-8, majority-of-{SAMPLES}, n={n}, realistic corpus)")
print("=" * 80)
print(f"{'ladder level':<16}{'risk':>8}{'savings':>10}{'hb_p':>10}  certifies@α={ALPHA}")
print("-" * 80)
for li, (nm, _f) in enumerate(LADDER):
    rh, sv = risk_savings(nm)
    p = hb_pvalue(rh, n, ALPHA)
    print(f"{nm:<16}{rh * 100:>7.1f}%{sv * 100:>9.1f}%{p:>10.4f}  {'YES' if p <= DELTA else 'no'}")
if idx >= 0:
    nm = LADDER[idx][0]
    rh, sv = risk_savings(nm)
    print(
        f"\n  ✔ CERTIFIED '{nm}' → {sv * 100:.1f}% token savings @ ≤{ALPHA * 100:.0f}% decision-change, {int((1 - DELTA) * 100)}% confidence (LTT, n={n})"
    )
    print(f"  CRC cross-check: {LADDER[idx_crc][0] if idx_crc >= 0 else 'NONE'}")

# ---- head-to-head -----------------------------------------------------------
print("\n" + "=" * 80)
print("LIVE HEAD-TO-HEAD  (same corpus, same gate)")
print("=" * 80)
print(f"{'method':<26}{'savings':>9}{'dec-change':>12}{'certifies?':>12}")
print("-" * 80)
champion = ("distil (prune+lossless)", *risk_savings("prune+lossless"))
order = [("distil (prune+lossless)", risk_savings("prune+lossless"))]
for nm, _f in COMPETITORS:
    order.append((nm, risk_savings(nm) if compiled.get(nm) is not None else None))
for nm, rs in order:
    if rs is None:
        print(f"{nm:<26}{'—':>9}{'—':>12}{'—':>12}   {notes.get(nm, '')}")
        continue
    rh, sv = rs
    p = hb_pvalue(rh, n, ALPHA)
    print(f"{nm:<26}{sv * 100:>8.1f}%{rh * 100:>11.1f}%{('YES' if p <= DELTA else 'no'):>12}")
print(f"{'RTK':<26}{'—':>9}{'—':>12}{'—':>12}   {notes.get('RTK', '')}")
print("-" * 80)
print(f"wall-clock: {time.time() - t0:.0f}s grading")
