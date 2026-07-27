"""Prometheus text-exposition metrics — stdlib only.

Enterprises scrape; they do not poll a dashboard. distil already computes every
number this module exposes (the gateway's per-tenant snapshot, the local savings
ledger); the gap was purely that there was no `/metrics` for Prometheus, Grafana
Agent, the OTel Collector, or a Datadog OpenMetrics check to read.

No client library: the exposition format is a few lines of text, and the core is
stdlib-only by design (pyproject.toml). Writing it by hand costs less than the
dependency would, and an exporter that cannot fail to import cannot take the
proxy down with it.

Naming follows Prometheus convention — `distil_` prefix, base units (tokens,
dollars, seconds), `_total` on monotonic counters — so the standard `rate()` and
`increase()` queries work without relabeling.
"""

from __future__ import annotations

from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# name -> (type, help). Kept in one place so /metrics and the docs cannot drift.
_SPEC: tuple[tuple[str, str, str], ...] = (
    ("distil_requests_total", "counter", "Requests processed, by tenant."),
    (
        "distil_tokens_baseline_total",
        "counter",
        "Input tokens that would have been sent uncompressed.",
    ),
    ("distil_tokens_sent_total", "counter", "Input tokens actually sent after compression."),
    ("distil_tokens_saved_total", "counter", "Input tokens kept off the wire."),
    (
        "distil_dollars_saved_total",
        "counter",
        "Value of saved tokens at the configured input price.",
    ),
    ("distil_compression_ratio", "gauge", "Fraction of input tokens saved (0-1)."),
    ("distil_build_info", "gauge", "Build metadata; always 1, carries the version label."),
)


def _escape(v: str) -> str:
    """Escape a label VALUE per the exposition format (backslash, quote, newline)."""
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _line(name: str, value: float, **labels: str) -> str:
    if labels:
        inner = ",".join(f'{k}="{_escape(str(v))}"' for k, v in sorted(labels.items()))
        name = f"{name}{{{inner}}}"
    # Integers render without a trailing .0 so counters read naturally in Grafana.
    text = str(int(value)) if float(value).is_integer() else repr(float(value))
    return f"{name} {text}"


def render(snapshot: dict[str, Any], *, version: str = "") -> str:
    """Render a gateway snapshot as Prometheus text exposition.

    `snapshot` is `TenantStats.snapshot()` — per-tenant rows plus totals. Totals
    are NOT emitted as separate series: Prometheus sums labelled series itself,
    and shipping both invites double-counting in a naive `sum()` query.
    """
    out: list[str] = []
    for name, kind, helptext in _SPEC:
        out.append(f"# HELP {name} {helptext}")
        out.append(f"# TYPE {name} {kind}")
        if name == "distil_build_info":
            out.append(_line(name, 1, version=version or "unknown"))
            continue
        for row in snapshot.get("tenants") or []:
            tenant = str(row.get("tenant", "unknown"))
            if name == "distil_requests_total":
                out.append(_line(name, row.get("requests", 0), tenant=tenant))
            elif name == "distil_tokens_baseline_total":
                out.append(_line(name, row.get("tokens_baseline", 0), tenant=tenant))
            elif name == "distil_tokens_sent_total":
                out.append(_line(name, row.get("tokens_compressed", 0), tenant=tenant))
            elif name == "distil_tokens_saved_total":
                out.append(_line(name, row.get("tokens_saved", 0), tenant=tenant))
            elif name == "distil_dollars_saved_total":
                out.append(_line(name, row.get("dollars_saved", 0.0), tenant=tenant))
            elif name == "distil_compression_ratio":
                # pct_saved is 0-100 on the wire; Prometheus convention is a ratio.
                out.append(_line(name, round(row.get("pct_saved", 0.0) / 100.0, 6), tenant=tenant))
    return "\n".join(out) + "\n"
