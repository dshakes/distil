"""distil.otel — GenAI spans are strictly optional and must never affect the
request path, with or without opentelemetry-sdk installed.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distil import otel
from distil.proxy import build_handler


def _payload() -> bytes:
    return json.dumps(
        {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()


def _start_echo_upstream() -> ThreadingHTTPServer:
    class Echo(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — stdlib handler method name
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            body = b'{"id": "msg_1", "content": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Echo)

    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_request_span_is_noop_when_disabled(monkeypatch):
    """Simulates opentelemetry-api not being importable at all."""
    monkeypatch.setattr(otel, "_ENABLED", False)
    with otel.request_span("claude-opus-4-8", "/v1/messages") as span:
        assert span is None
    otel.set_result_attrs(span, original_tokens=10, compressed_tokens=5, compressed=True)


def test_proxy_round_trip_unaffected_when_otel_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(otel, "_ENABLED", False)
    upstream = _start_echo_upstream()
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1])
        conn.request(
            "POST", "/v1/messages", body=_payload(), headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
    finally:
        proxy.shutdown()
        upstream.shutdown()


def test_request_span_records_attrs_with_sdk(monkeypatch):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel, "_ENABLED", True)
    monkeypatch.setattr(otel, "_tracer", provider.get_tracer("test"))
    monkeypatch.setenv("DISTIL_SESSION", "s123-42")

    with otel.request_span("claude-opus-4-8", "/v1/messages") as span:
        otel.set_result_attrs(
            span,
            original_tokens=120,
            compressed_tokens=40,
            compression_ratio=0.6,
            compressed=True,
            shadow_sampled=False,
        )

    (finished,) = exporter.get_finished_spans()
    assert finished.name == "chat claude-opus-4-8"
    assert finished.attributes["gen_ai.request.model"] == "claude-opus-4-8"
    assert finished.attributes["gen_ai.system"] == "anthropic"
    # input_tokens = the prompt actually sent upstream (compressed count);
    # output_tokens is never set here — it means *generated* tokens
    assert finished.attributes["gen_ai.usage.input_tokens"] == 40
    assert finished.attributes["distil.tokens.original"] == 120
    assert finished.attributes["distil.tokens.compressed"] == 40
    assert "gen_ai.usage.output_tokens" not in finished.attributes
    assert finished.attributes["distil.compression.applied"] is True
    assert finished.attributes["distil.compression.ratio"] == 0.6
    assert finished.attributes["distil.shadow.sampled"] is False
    # the same id the ledger/proof-ledger key on — cross-signal correlation
    assert finished.attributes["distil.session.id"] == "s123-42"


def test_proxy_round_trip_emits_span_with_sdk(monkeypatch):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel, "_ENABLED", True)
    monkeypatch.setattr(otel, "_tracer", provider.get_tracer("test"))

    upstream = _start_echo_upstream()
    handler = build_handler(f"http://127.0.0.1:{upstream.server_address[1]}")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1])
        conn.request(
            "POST", "/v1/messages", body=_payload(), headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
    finally:
        proxy.shutdown()
        upstream.shutdown()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.request.model"] == "claude-opus-4-8"


def test_otel_failures_never_break_the_request_path(monkeypatch):
    """Every guard in the module: a tracer that explodes at span start, a span
    that explodes on set_attribute/end — the caller must never see any of it."""
    from distil import otel

    class _Bomb:
        def start_as_current_span(self, *a, **k):
            raise RuntimeError("exporter down")

    monkeypatch.setattr(otel, "_ENABLED", True)
    monkeypatch.setattr(otel, "_tracer", _Bomb())
    with otel.request_span("claude-opus-4-8", "/v1/messages") as span:
        assert span is None  # start failed silently → no-op span
        otel.set_result_attrs(span, original_tokens=1)  # None-span: no-op

    class _BadSpan:
        def set_attribute(self, *a):
            raise RuntimeError("attr")

    class _BadCM:
        def __enter__(self):
            return _BadSpan()

        def __exit__(self, *a):
            raise RuntimeError("exit")

    class _BadTracer:
        def start_as_current_span(self, *a, **k):
            return _BadCM()

    monkeypatch.setattr(otel, "_tracer", _BadTracer())
    with otel.request_span("claude-opus-4-8", "/v1/messages") as span:
        assert span is not None  # started, but every method on it explodes
        otel.set_result_attrs(span, original_tokens=2, compressed=True)  # swallowed
    # reaching here without an exception IS the assertion


# ---------------------------------------------------------------------------
# OTel metrics — the counter half of the export. Independent of tracing.
# ---------------------------------------------------------------------------


def test_record_compression_is_a_silent_noop_without_the_sdk(monkeypatch) -> None:
    """The core has no runtime dependencies, so the common case is no metrics
    SDK at all. That path must cost nothing and never raise."""
    monkeypatch.setattr(otel, "_METRICS_ENABLED", False)
    otel.record_compression(1000, 400, model="claude-opus-4-8")  # must not raise


def test_record_compression_never_raises_on_a_broken_instrument(monkeypatch) -> None:
    """A misbehaving exporter must not take down the request it is measuring."""

    class _Boom:
        def add(self, *a, **k):
            raise RuntimeError("collector exploded")

    monkeypatch.setattr(otel, "_METRICS_ENABLED", True)
    monkeypatch.setattr(otel, "_c_requests", _Boom())
    otel.record_compression(1000, 400)  # swallowed


def test_counters_export_the_real_numbers() -> None:
    """Assert the VALUES an OTel collector would receive, not merely that the
    call succeeded — a metrics export that reports zeros is worse than none."""
    sdk = pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = sdk.MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("distil-test")

    import distil.otel as o

    saved_state = (o._METRICS_ENABLED, o._c_requests, o._c_baseline, o._c_sent, o._c_saved)
    try:
        o._METRICS_ENABLED = True
        o._c_requests = meter.create_counter("distil.requests")
        o._c_baseline = meter.create_counter("distil.tokens.baseline")
        o._c_sent = meter.create_counter("distil.tokens.sent")
        o._c_saved = meter.create_counter("distil.tokens.saved")

        o.record_compression(1000, 400, model="claude-opus-4-8", provider="anthropic")
        o.record_compression(500, 200, model="claude-opus-4-8", provider="anthropic")

        totals: dict[str, float] = {}
        for rm in reader.get_metrics_data().resource_metrics or []:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    for pt in m.data.data_points:
                        totals[m.name] = totals.get(m.name, 0) + pt.value
    finally:
        (o._METRICS_ENABLED, o._c_requests, o._c_baseline, o._c_sent, o._c_saved) = saved_state

    assert totals["distil.requests"] == 2
    assert totals["distil.tokens.baseline"] == 1500
    assert totals["distil.tokens.sent"] == 600
    assert totals["distil.tokens.saved"] == 900, "saved must equal baseline - sent"


def test_metrics_are_recorded_even_when_tracing_is_off(monkeypatch) -> None:
    """Tracing and metrics are independently optional. set_result_attrs() with a
    None span (tracer absent or sampled out) must STILL record the counters —
    otherwise metrics silently depend on a tracer nobody installed."""
    seen: list[tuple] = []
    monkeypatch.setattr(otel, "record_compression", lambda o_, c_, **k: seen.append((o_, c_)))
    otel.set_result_attrs(None, original_tokens=900, compressed_tokens=300)
    assert seen == [(900, 300)], "counters were skipped when the span was None"
