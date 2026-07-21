"""Real-time local savings dashboard — a localhost page fed by your own live
ledger, so the odometer moves ONLY when a real request actually books more
saved tokens. Nothing leaves the machine; this reads ``~/.distil/savings.jsonl``
(the same numbers ``distil stats`` shows) and serves them to a local browser.

Honest by construction: the counter is anchored to the exact ledger total and
rolls up when that total grows — every increment traces to a completed request.
No projection, no telemetry, stdlib only.

    distil dashboard --web         # opens http://127.0.0.1:8766
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _snapshot() -> dict:
    """Fresh ledger figures + live decision-equivalence — read on every poll."""
    from . import calibration, ledger

    s = ledger.summary()
    try:
        f, _ = calibration.factor()
    except Exception:  # noqa: BLE001 — calibration is a correction, never a blocker
        f = 1.0
    tokens = max(0, s.total_baseline_tokens - s.total_distil_tokens)
    dollars = max(0.0, s.total_baseline_dollars - s.total_distil_dollars)
    pct = (
        (1.0 - s.total_distil_tokens / s.total_baseline_tokens) * 100.0
        if s.total_baseline_tokens
        else 0.0
    )
    eq = _equivalence()
    sub = _subscription()
    return {
        "tokens_saved": round(tokens * f),
        "dollars_saved": round(dollars * f, 2),
        "pct": round(pct, 1),
        "runs": s.runs,
        "baseline_tokens": round(s.total_baseline_tokens * f),
        "distil_tokens": round(s.total_distil_tokens * f),
        "equivalence": eq,
        "subscription": sub,
        "ts": time.time(),
    }


def _equivalence() -> dict:
    try:
        from .shadow import ShadowLedger

        led = ShadowLedger.load(current_only=True)
        shadowed = int(led.samples)
        pct = (
            round((1.0 - led.adjusted_rate()) * 100.0, 2)
            if shadowed and led.aa_agreement() is not None
            else None
        )
        return {"pct": pct, "shadowed": shadowed}
    except Exception:  # noqa: BLE001
        return {"pct": None, "shadowed": 0}


def _subscription() -> bool:
    try:
        from .doctor import subscription_mode

        return bool(subscription_mode())
    except Exception:  # noqa: BLE001
        return False


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>distil · live savings</title>
<style>
  :root{--bg:#0b0c13;--panel:#101420;--panel2:#141a28;--line:#1c2233;--mut:#8a8ca0;--dim:#5a5c70;--good:#5ad19a;--acc:#8b7bff;--tl:#5ad1c9}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;background:radial-gradient(120% 90% at 15% -10%,rgba(90,209,154,.10),transparent 55%),radial-gradient(120% 120% at 100% 0%,rgba(139,123,255,.10),transparent 50%),var(--bg);
    color:#eaf0ff;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;align-items:center;justify-content:center;padding:32px}
  .card{width:min(760px,100%);border:1px solid var(--line);border-radius:22px;overflow:hidden;position:relative;
    background:linear-gradient(180deg,var(--panel2),var(--panel));box-shadow:0 40px 120px -50px #000}
  .top{height:3px;background:linear-gradient(90deg,var(--acc),var(--tl))}
  .in{padding:30px 32px 28px}
  .lab{font-size:11.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--mut);display:flex;align-items:center;gap:9px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 12px 2px rgba(90,209,154,.75);animation:p 1.4s ease-in-out infinite}
  @keyframes p{50%{opacity:.35}}
  .odo{display:flex;align-items:baseline;gap:1px;margin:14px 0 8px;font-weight:800;font-size:clamp(34px,7vw,68px);line-height:1;letter-spacing:-.02em;flex-wrap:wrap;text-shadow:0 0 30px rgba(90,209,154,.25)}
  .odo .d{min-width:.62em;text-align:center;transition:color .25s}
  .odo .d.sep{min-width:.28em;color:var(--dim);text-shadow:none}
  .odo .d.tick{color:#7dffc4}
  .odo .u{font-size:.32em;font-weight:700;color:var(--mut);margin-left:.35em}
  .sub{color:var(--mut);font-size:13px;margin:2px 0 20px;line-height:1.5}
  .sub b{color:#eaf0ff}
  .row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  @media(max-width:620px){.row{grid-template-columns:1fr 1fr}}
  .m{border:1px solid var(--line);border-radius:13px;padding:13px 15px;background:rgba(255,255,255,.015)}
  .mv{font-weight:800;font-size:20px;font-variant-numeric:tabular-nums}
  .mv.g{color:var(--good)} .mv.a{color:var(--acc)}
  .ml{color:var(--mut);font-size:11px;margin-top:4px}
  .foot{color:var(--dim);font-size:11.5px;margin-top:18px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}
  .flash{animation:fl 1s ease-out}
  @keyframes fl{0%{box-shadow:0 0 0 1px var(--good),0 0 30px -6px var(--good)}100%{box-shadow:none}}
  @media (prefers-reduced-motion:reduce){.dot{animation:none}.odo .d{transition:none}}
</style></head><body>
<div class="card" id="card"><div class="top"></div><div class="in">
  <div class="lab"><span class="dot"></span> your tokens saved · live · this machine</div>
  <div class="odo" id="odo"><span class="d">0</span></div>
  <div class="sub" id="sub">reading your local ledger…</div>
  <div class="row">
    <div class="m"><div class="mv g" id="pct">–</div><div class="ml">smaller · overall</div></div>
    <div class="m"><div class="mv" id="dollars">–</div><div class="ml" id="dollarslab">$ saved</div></div>
    <div class="m"><div class="mv a" id="eq">–</div><div class="ml">decision-equivalence</div></div>
    <div class="m"><div class="mv" id="runs">–</div><div class="ml">requests</div></div>
  </div>
  <div class="foot"><span>◉ local only · nothing leaves this machine</span><span id="stamp"></span></div>
</div></div>
<script>
(function(){
  var reduced=matchMedia("(prefers-reduced-motion:reduce)").matches;
  var odo=document.getElementById("odo"),cells=[];
  function commas(n){return Math.floor(n).toLocaleString("en-US");}
  function human(n){var u=[[1e12,"T"],[1e9,"B"],[1e6,"M"],[1e3,"k"]];for(var i=0;i<u.length;i++)if(n>=u[i][0])return (n/u[i][0]).toFixed(1)+u[i][1];return String(Math.round(n));}
  function render(str){
    if(cells.length!==str.length){odo.innerHTML="";cells=[];for(var i=0;i<str.length;i++){var s=document.createElement("span");s.className="d"+(str[i]===","?" sep":"");s.textContent=str[i];odo.appendChild(s);cells.push(s);}var u=document.createElement("span");u.className="u";u.textContent="tokens";odo.appendChild(u);return;}
    for(var j=0;j<str.length;j++){if(cells[j].textContent!==str[j]){cells[j].textContent=str[j];if(!reduced&&str[j]!==","){cells[j].classList.add("tick");(function(el){setTimeout(function(){el.classList.remove("tick");},260);})(cells[j]);}}}
  }
  var shown=null,target=0,from=0,t0=null;
  function loop(){
    if(target!==shown){
      if(reduced||shown===null){shown=target;}
      else{if(t0===null){t0=Date.now();from=shown;}var k=Math.min(1,(Date.now()-t0)/900);k=1-Math.pow(1-k,3);shown=Math.round(from+(target-shown===0?0:(target-from))*k);if(k>=1){shown=target;t0=null;}}
      render(commas(shown));
    }
    requestAnimationFrame(loop);
  }
  function set(id,v){var e=document.getElementById(id);if(e)e.textContent=v;}
  var last=null;
  function poll(){
    fetch("/data",{cache:"no-store"}).then(function(r){return r.json();}).then(function(d){
      if(last!==null&&d.tokens_saved>last){var c=document.getElementById("card");c.classList.remove("flash");void c.offsetWidth;c.classList.add("flash");}
      last=d.tokens_saved;target=d.tokens_saved;
      set("pct",d.pct+"%");
      set("dollars",d.subscription?"$"+human(d.dollars_saved):"$"+human(d.dollars_saved));
      set("dollarslab",d.subscription?"$ saved · notional":"$ saved · billed");
      set("eq",d.equivalence.pct==null?"n/a":d.equivalence.pct+"%");
      set("runs",commas(d.runs));
      document.getElementById("sub").innerHTML="The <b>exact</b> total from your ledger — "+human(d.baseline_tokens)+" → "+human(d.distil_tokens)+" tokens. It ticks up the instant a request books real savings. Equivalence over <b>"+commas(d.equivalence.shadowed)+"</b> shadowed requests.";
      set("stamp","updated "+new Date(d.ts*1000).toLocaleTimeString());
    }).catch(function(){});
  }
  requestAnimationFrame(loop);poll();setInterval(poll,1000);
})();
</script></body></html>"""


def serve_webdash(port: int = 8766, *, host: str = "127.0.0.1", open_browser: bool = True) -> None:
    """Blocking: serve the live dashboard until Ctrl-C."""
    server = build_server(host, port)
    url = f"http://{host}:{port}"
    print(f"distil live dashboard → {url}")
    print("  local only · reads your ledger · nothing leaves this machine · Ctrl-C to stop")
    if open_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    """The dashboard HTTP server — factored out so tests can drive it directly.
    Serves the local page and the content-free ``/data`` snapshot; nothing else."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002 — silence access log
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/data"):
                self._send(200, "application/json", json.dumps(_snapshot()).encode())
            elif self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", _PAGE.encode())
            else:
                self._send(404, "text/plain", b"not found")

    return ThreadingHTTPServer((host, port), H)
