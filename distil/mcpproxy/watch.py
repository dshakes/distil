"""See how ``distil mcp`` compressed: ``distil mcp watch`` (terminal) and webdash ``/mcp``.

Both read the same content-free local files (``events``) and render the same
``summarize`` output, so the two views can never disagree. Tool names appear here —
they are local-only and never reach the census or telemetry.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import IO, Any

from . import events, levels

CERT_PATH = Path(__file__).resolve().parent.parent / "certificates" / "mcp.json"


def certificate(path: Path | None = None) -> dict[str, str]:
    """Per-level accuracy-certificate status: ``pending`` | ``certified`` | ``failed``."""
    try:
        data = json.loads((path or CERT_PATH).read_text(encoding="utf-8"))
        raw = data.get("levels") or {}
    except (OSError, ValueError, AttributeError):
        raw = {}
    out = {}
    for lv in (*levels.LEVELS, "R"):
        entry = raw.get(lv)
        status = entry.get("status") if isinstance(entry, dict) else None
        out[lv] = status if status in ("pending", "certified", "failed") else "pending"
    return out


def summarize(
    rows: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    cert: dict[str, str],
) -> dict[str, Any]:
    """Aggregate the event log into per-server and per-tool figures, latest session first."""
    by_server: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if isinstance(r.get("server"), str) and r["server"] != "*":
            by_server.setdefault(r["server"], []).append(r)
    servers = []
    for name in sorted(set(by_server) | set(catalogs)):
        evs = by_server.get(name, [])
        cat = catalogs.get(name, {})
        session = (
            max(evs, key=lambda r: r.get("ts", 0)).get("session") if evs else cat.get("session")
        )
        cur = [r for r in evs if r.get("session") == session]
        lists = [r for r in cur if r.get("ev") == "list"]
        last_list = lists[-1] if lists else {}
        unlocked = [r.get("tool") for r in cur if r.get("ev") == "unlock"]
        fetches = Counter(r.get("tool") for r in cur if r.get("ev") == "schema_fetch")
        calls = [r for r in cur if r.get("ev") == "call"]
        ncalls = Counter(r.get("tool") for r in calls)
        rb: Counter[str] = Counter()
        ra: Counter[str] = Counter()
        for r in calls:
            rb[r.get("tool") or ""] += int(r.get("tokens_before") or 0)
            ra[r.get("tool") or ""] += int(r.get("tokens_after") or 0)
        lazy = cat.get("level") in ("L2", "L3")
        tools = []
        for t in cat.get("tools") or []:
            n = t.get("name")
            if t.get("pinned"):
                state = "pinned"
            elif lazy and n in unlocked:
                state = "unlocked"
            elif lazy:
                state = "lazy"
            else:
                state = "full"
            after = t.get("tokens_lazy") if state == "lazy" else t.get("tokens_after")
            tools.append(
                {
                    "name": n,
                    "state": state,
                    "def_before": t.get("tokens_before", 0),
                    "def_after": after or 0,
                    "dropped": len(t.get("dropped") or []),
                    "schema_fetches": fetches.get(n, 0),
                    "calls": ncalls.get(n, 0),
                    "result_before": rb.get(n, 0),
                    "result_after": ra.get(n, 0),
                }
            )
        servers.append(
            {
                "server": name,
                "session": session,
                "level": cat.get("level") or last_list.get("level") or "?",
                "requested": cat.get("requested") or cat.get("level") or "?",
                "results": bool(cat.get("results", True)),
                "defs_before": int(
                    last_list.get("tokens_before") or sum(t["def_before"] for t in tools)
                ),
                "defs_after": int(
                    last_list.get("tokens_after") or sum(t["def_after"] for t in tools)
                ),
                "list_changes": sum(1 for r in cur if r.get("ev") in ("unlock", "list_changed")),
                "schema_fetches": sum(fetches.values()),
                "calls": len(calls),
                "result_before": sum(rb.values()),
                "result_after": sum(ra.values()),
                "expands": sum(1 for r in cur if r.get("ev") == "expand"),
                "errors": sum(1 for r in cur if r.get("ev") == "error"),
                "tools": tools,
            }
        )
    timeline = [
        {
            k: r.get(k)
            for k in ("ts", "server", "ev", "tool", "tokens_before", "tokens_after")
            if r.get(k) is not None
        }
        for r in rows[-60:]
    ]
    return {"servers": servers, "certificate": cert, "timeline": timeline[::-1], "ts": time.time()}


def load_summary(root: Path | None = None) -> dict[str, Any]:
    base = root or events.mcp_dir()
    return summarize(
        events.read_events(base / "events.jsonl"), events.read_catalogs(base), certificate()
    )


def _pct(before: int, after: int) -> str:
    if not before:
        return "—"
    p = 100.0 * (before - after) / before
    return "0%" if p <= 0 else ("<1%" if p < 1 else f"{p:.0f}%")


def render_text(summary: dict[str, Any], *, width: int = 100, max_tools: int = 25) -> str:
    cert = summary.get("certificate") or {}
    lines = [
        "distil mcp watch — local only, tool names never leave this machine",
        "certificate  "
        + "  ".join(f"{lv}:{cert.get(lv, 'pending')}" for lv in (*levels.LEVELS, "R")),
        "",
    ]
    if not summary.get("servers"):
        lines.append("no MCP traffic yet — route a server through `distil mcp wrap` and use it")
    for s in summary.get("servers", []):
        level = (
            s["level"]
            if s["level"] == s["requested"]
            else f"{s['level']} (asked {s['requested']}; not smaller)"
        )
        cache = "stable" if s["list_changes"] == 0 else f"{s['list_changes']} list change(s)"
        lines.append(
            f"[{s['server']}] level {level} ({levels.LEVEL_NAMES.get(s['level'], '?')})"
            f"  results {'on' if s['results'] else 'off'}  cache {cache}"
        )
        lines.append(
            f"  definitions {s['defs_before']:,} → {s['defs_after']:,} tok ({_pct(s['defs_before'], s['defs_after'])} smaller)"
            f"  schema fetches {s['schema_fetches']}"
        )
        lines.append(
            f"  results {s['result_before']:,} → {s['result_after']:,} tok"
            f" ({_pct(s['result_before'], s['result_after'])} smaller)  calls {s['calls']}"
            f"  expands {s['expands']}" + (f"  errors {s['errors']}" if s["errors"] else "")
        )
        lines.append(
            f"  {'tool':<34}{'state':<10}{'def tok':>16}{'fetch':>7}{'calls':>7}{'result tok':>18}"
        )
        tools = sorted(
            s["tools"], key=lambda t: (-t["calls"], -t["schema_fetches"], t["name"] or "")
        )
        for t in tools[:max_tools]:
            res = f"{t['result_before']:,}→{t['result_after']:,}" if t["result_before"] else "—"
            lines.append(
                f"  {str(t['name'])[:33]:<34}{t['state']:<10}{t['def_before']:>7,}→{t['def_after']:<7,}"
                f"{t['schema_fetches']:>7}{t['calls']:>7}{res:>18}"
            )
        if len(tools) > max_tools:
            lines.append(
                f"  … {len(tools) - max_tools} more (distil dashboard --web → /mcp for all)"
            )
        lines.append("")
    return "\n".join(ln[:width] if len(ln) > width else ln for ln in lines)


def run_watch(
    *,
    interval: float = 1.0,
    once: bool = False,
    stream: IO[str] | None = None,
    root: Path | None = None,
) -> int:
    out = stream or sys.stdout
    try:
        while True:
            text = render_text(load_summary(root))
            if once:
                out.write(text + "\n")
                return 0
            out.write("\x1b[H\x1b[2J" + text + "\n")
            out.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def tool_detail(server: str, tool: str, root: Path | None = None) -> dict[str, Any] | None:
    """Before/after model-facing definition of one tool, for the webdash diff view."""
    cat = events.read_catalogs(root).get(server)
    for t in (cat or {}).get("tools") or []:
        if t.get("name") == tool:
            return {
                "server": server,
                "name": tool,
                "level": cat.get("level") if cat else None,
                "before": t.get("before"),
                "after": t.get("after"),
                "dropped": t.get("dropped") or [],
                "recoverable": "full definition via "
                + levels.meta_name(server, "get_tool_schema")
                + (
                    " (and the server itself)"
                    if cat and cat.get("level") != "L0"
                    else "; L0 drops only annotation keywords"
                ),
            }
    return None


# The /mcp page. Every server-provided string (tool names, schemas) is inserted with
# textContent — a backend controls those strings, so none of them may reach innerHTML.
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>distil · MCP compression</title>
<style>
  :root{--bg:#0b0c13;--panel:#101420;--line:#1c2233;--mut:#8a8ca0;--good:#5ad19a;--acc:#8b7bff;--bad:#ff7b7b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:#eaf0ff;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;padding:24px;line-height:1.45}
  main{max-width:1100px;margin:0 auto}
  h1{font-size:18px;margin:0 0 6px} h2{font-size:15px;margin:22px 0 8px}
  p,.mut{color:var(--mut);font-size:13px}
  table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 10px}
  caption{text-align:left;color:var(--mut);padding:4px 0}
  th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right}
  th:first-child,td:first-child,td.l,th.l{text-align:left}
  .pill{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:1px 7px;margin-right:6px;font-size:12px}
  .certified{color:var(--good)} .failed{color:var(--bad)} .pending{color:var(--mut)}
  button{background:transparent;border:1px solid var(--line);color:#eaf0ff;font:inherit;font-size:12px;padding:3px 9px;border-radius:6px;cursor:pointer}
  button:focus-visible,a:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
  .diff{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  @media(max-width:760px){.diff{grid-template-columns:1fr}}
  pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;max-height:420px;font-size:12px;white-space:pre-wrap}
  ol{font-size:12.5px;padding-left:22px}
  .sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
  @media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style></head><body><main>
<h1>MCP compression · live · this machine</h1>
<p>Definitions and results compressed by <code>distil mcp</code>. Read from the local, content-free event log; tool names never leave this machine.</p>
<p id="cert" aria-label="accuracy certificate status per level"></p>
<p id="live" class="sr-only" role="status"></p>
<div id="servers"></div>
<section aria-labelledby="diffh"><h2 id="diffh">Before / after</h2>
<p id="diffhint" class="mut">Choose a tool's “diff” button to compare what the model is sent.</p>
<div id="diffmeta"></div>
<div class="diff"><div><h3 class="mut">original</h3><pre id="before" tabindex="0" aria-label="original definition"></pre></div>
<div><h3 class="mut">sent</h3><pre id="after" tabindex="0" aria-label="compressed definition"></pre></div></div>
<ul id="dropped" class="mut"></ul></section>
<section aria-labelledby="tlh"><h2 id="tlh">Timeline</h2><ol id="timeline"></ol></section>
<button type="button" id="pause" aria-pressed="false">Pause</button>
</main>
<script>
(function(){
  var paused=false,timer=null,last="";
  function el(tag,text,cls){var e=document.createElement(tag);if(text!=null)e.textContent=String(text);if(cls)e.className=cls;return e;}
  function pct(b,a){if(!b)return "—";var p=100*(b-a)/b;return p<=0?"0%":(p<1?"<1%":Math.round(p)+"%");}
  function n(x){return Number(x||0).toLocaleString("en-US");}
  function diff(server,tool){
    fetch("/mcp/tool?server="+encodeURIComponent(server)+"&tool="+encodeURIComponent(tool),{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
      document.getElementById("before").textContent=JSON.stringify(d.before,null,1);
      document.getElementById("after").textContent=JSON.stringify(d.after,null,1);
      document.getElementById("diffmeta").textContent=server+" / "+tool+" · level "+d.level+" · recoverable: "+d.recoverable;
      var ul=document.getElementById("dropped");ul.textContent="";
      (d.dropped||[]).forEach(function(p){ul.appendChild(el("li",p[1]+" "+p[0]));});
      if(!(d.dropped||[]).length)ul.appendChild(el("li","nothing removed or shortened"));
      document.getElementById("diffhint").textContent="Showing "+tool+".";
    }).catch(function(){});
  }
  function render(d){
    var c=document.getElementById("cert");c.textContent="accuracy certificate: ";
    Object.keys(d.certificate||{}).forEach(function(k){c.appendChild(el("span",k+" "+d.certificate[k],"pill "+d.certificate[k]));});
    var root=document.getElementById("servers");root.textContent="";
    (d.servers||[]).forEach(function(s){
      root.appendChild(el("h2",s.server+" · "+s.level+(s.level!==s.requested?" (asked "+s.requested+", not smaller)":"")));
      root.appendChild(el("p","definitions "+n(s.defs_before)+" → "+n(s.defs_after)+" tokens ("+pct(s.defs_before,s.defs_after)+" smaller) · results "+n(s.result_before)+" → "+n(s.result_after)+" · schema fetches "+s.schema_fetches+" · expands "+s.expands+" · cache "+(s.list_changes?s.list_changes+" list change(s)":"stable"),"mut"));
      var t=el("table");var cap=el("caption","Per-tool compression for "+s.server);t.appendChild(cap);
      var hr=el("tr");["tool","state","def before","def after","fetches","calls","result before","result after",""].forEach(function(h,i){var th=el("th",h);th.scope="col";if(i<2)th.className="l";hr.appendChild(th);});
      var th=el("thead");th.appendChild(hr);t.appendChild(th);var tb=el("tbody");
      s.tools.forEach(function(x){var tr=el("tr");var th=el("th",x.name,"l");th.scope="row";tr.appendChild(th);
        [x.state,n(x.def_before),n(x.def_after),x.schema_fetches,x.calls,n(x.result_before),n(x.result_after)].forEach(function(v,i){tr.appendChild(el("td",v,i===0?"l":null));});
        var td=el("td");var b=el("button","diff");b.type="button";b.setAttribute("aria-label","show before and after for "+x.name);b.addEventListener("click",function(){diff(s.server,x.name);});td.appendChild(b);tr.appendChild(td);tb.appendChild(tr);});
      t.appendChild(tb);root.appendChild(t);
    });
    if(!(d.servers||[]).length)root.appendChild(el("p","No MCP traffic yet — route a server through distil mcp wrap and use it.","mut"));
    var ol=document.getElementById("timeline");ol.textContent="";
    (d.timeline||[]).forEach(function(r){ol.appendChild(el("li",new Date(r.ts*1000).toLocaleTimeString()+" "+r.server+" "+r.ev+(r.tool?" "+r.tool:"")+(r.tokens_before!=null?" "+n(r.tokens_before)+"→"+n(r.tokens_after)+" tok":"")));});
  }
  function poll(){fetch("/mcp/data",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
    var key=JSON.stringify(d.servers)+JSON.stringify(d.timeline);if(key===last)return;last=key;render(d);
    document.getElementById("live").textContent=(d.servers||[]).length+" MCP server(s) updated";}).catch(function(){});}
  document.getElementById("pause").addEventListener("click",function(){paused=!paused;this.textContent=paused?"Resume":"Pause";this.setAttribute("aria-pressed",String(paused));if(paused){clearInterval(timer);}else{poll();timer=setInterval(poll,1500);}});
  poll();timer=setInterval(poll,1500);
})();
</script></body></html>"""
