import sys, json, re, copy, hashlib

sys.path.insert(0, "benchmarks")
import model_migration_eval as M
from distil.adapters import anthropic as A

NUM = re.compile(r"^(\d+):")


def window(text, cmd, name):
    m = re.match(r"(\d+)(?::(\d+))?", cmd.split(" ", 1)[1] if " " in cmd else "")
    if name == "edit" and m:
        lo, hi = int(m.group(1)) - 1, int(m.group(2) or m.group(1)) + 1
    elif name == "goto" and m:
        lo, hi = int(m.group(1)) - 2, int(m.group(1)) + 10
    else:
        return None
    out = []
    drop = 0
    h = hashlib.sha256(text.encode()).hexdigest()[:8]
    for ln in text.splitlines():
        mm = NUM.match(ln)
        if mm and not (lo <= int(mm.group(1)) <= hi):
            drop += 1
            continue
        if drop:
            out.append(f"<< +{drop} lines, handle={h} >>")
            drop = 0
        out.append(ln)
    if drop:
        out.append(f"<< +{drop} lines, handle={h} >>")
    return "\n".join(out)


orig = A.compress_messages


def patched(msgs, *a, **k):
    out, store = orig(msgs, *a, **k)
    tu = {}
    for m in msgs:
        for b in m["content"] if isinstance(m["content"], list) else []:
            if b.get("type") == "tool_use":
                tu[b["id"]] = (b["name"], b["input"]["command"])
    for mi, m in enumerate(out):
        if not isinstance(m["content"], list):
            continue
        for bi, b in enumerate(m["content"]):
            if b.get("type") != "tool_result":
                continue
            o = msgs[mi]["content"][bi]["content"]
            if not isinstance(o, str) or b["content"] == o:
                continue
            name, cmd = tu[b["tool_use_id"]]
            w = window(o, cmd, name) if "[File: " in o else None
            if w is not None and len(w) < len(o):
                b["content"] = w
    return out, store


A.compress_messages = patched
if __name__ == "__main__":
    S = json.load(open(".claude/hillclimb/compression-coding/_state.json"))
    if sys.argv[1] == "sav":
        files = sorted({t.rsplit("-t", 1)[0] for t in S["train_ids"]})
        full = served = 0
        for f in files:
            for _, _, req in M._swe_turns(M.SWE_DIR / (f + ".traj")):
                full += sum(
                    M.TOK.count(b.text) for b in M._swe_blocks(req, [o for _, o in req["pairs"]])
                )
                served += sum(M.TOK.count(b.text) for b in M.serve(req)[0])
        print("window", full, served, round(1 - served / full, 4))
    else:
        cs = {c["id"]: c for c in M.load_coding_cases()}
        req = cs[sys.argv[1]]["req"]
        obs = [b for b in M.serve(req)[0] if ":obs@" in b.id]
        print(obs[int(sys.argv[2])].text)
