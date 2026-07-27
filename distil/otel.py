"""Optional OpenTelemetry GenAI semantic-convention spans.

``opentelemetry-api`` is not a required dependency (see the stdlib-only core
constraint in pyproject.toml): every public function here degrades to a
no-op if it isn't importable, or if the tracer fails for any reason. Install
the ``otel`` extra to get real spans.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace

    _tracer: Any = trace.get_tracer("distil")
    _ENABLED = True
except Exception:  # noqa: BLE001 — observability must never break the request path
    _tracer = None
    _ENABLED = False

# Metrics are a separate optional surface from tracing: a fleet may run an OTel
# collector for metrics and scrape Prometheus for the same numbers, or neither.
# Instruments are created once and named to match distil/metrics.py so the two
# exports agree — one number, two transports.
try:
    from opentelemetry import metrics as _otel_metrics

    _meter: Any = _otel_metrics.get_meter("distil")
    _c_requests: Any = _meter.create_counter(
        "distil.requests", unit="{request}", description="Requests processed by distil."
    )
    _c_baseline: Any = _meter.create_counter(
        "distil.tokens.baseline", unit="{token}", description="Input tokens before compression."
    )
    _c_sent: Any = _meter.create_counter(
        "distil.tokens.sent", unit="{token}", description="Input tokens actually sent."
    )
    _c_saved: Any = _meter.create_counter(
        "distil.tokens.saved", unit="{token}", description="Input tokens kept off the wire."
    )
    _METRICS_ENABLED = True
except Exception:  # noqa: BLE001 — an absent metrics SDK must not break the request path
    _meter = None
    _c_requests = _c_baseline = _c_sent = _c_saved = None
    _METRICS_ENABLED = False


def record_compression(
    original_tokens: int | None,
    compressed_tokens: int | None,
    *,
    model: str = "",
    provider: str = "",
) -> None:
    """Record one request's compression on the OTel counters.

    Called from the same place the span attributes are set, so tracing and
    metrics can never disagree about a request. Silently does nothing when the
    metrics SDK is absent — which is the default, since the core has no runtime
    dependencies. Never raises.
    """
    if not _METRICS_ENABLED:
        return
    try:
        attrs = {k: v for k, v in (("model", model), ("provider", provider)) if v}
        _c_requests.add(1, attrs)
        if original_tokens is not None:
            _c_baseline.add(max(0, int(original_tokens)), attrs)
        if compressed_tokens is not None:
            _c_sent.add(max(0, int(compressed_tokens)), attrs)
        if original_tokens is not None and compressed_tokens is not None:
            _c_saved.add(max(0, int(original_tokens) - int(compressed_tokens)), attrs)
    except Exception:  # noqa: BLE001 — observability must never break the request path
        pass


def _provider_from_path(path: str) -> str:
    if "generateContent" in path or "countTokens" in path:
        return "gcp.gemini"
    if "chat/completions" in path or "/responses" in path:
        return "openai"
    return "anthropic"


@contextmanager
def request_span(model: str, path: str) -> Iterator[Any]:
    """Open a ``chat {model}`` span per the OTel GenAI semantic conventions.

    Yields ``None`` (a usable no-op) unless opentelemetry-api is installed
    and the tracer starts cleanly; a failure to start or close the span must
    never affect the request it wraps.
    """
    if not _ENABLED:
        yield None
        return
    try:
        span_cm = _tracer.start_as_current_span(f"chat {model}")
        span = span_cm.__enter__()
    except Exception:  # noqa: BLE001 — observability must never break the request path
        yield None
        return
    try:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", _provider_from_path(path))
        span.set_attribute("gen_ai.provider.name", _provider_from_path(path))
        span.set_attribute("gen_ai.request.model", model)
        # The identifier the rest of distil already keys on (ledger, proof ledger,
        # shadow) — lets a backend correlate every span of one wrap session.
        session = os.environ.get("DISTIL_SESSION", "")
        if session:
            span.set_attribute("distil.session.id", session)
    except Exception:  # noqa: BLE001 — observability must never break the request path
        pass
    try:
        yield span
    finally:
        # exc_info reflects the caller's in-flight exception, if any, so the
        # span records it — but a broken exporter must not mask that exception.
        try:
            span_cm.__exit__(*sys.exc_info())
        except Exception:  # noqa: BLE001 — observability must never break the request path
            pass


def set_result_attrs(
    span: Any,
    *,
    original_tokens: int | None = None,
    compressed_tokens: int | None = None,
    compression_ratio: float | None = None,
    compressed: bool | None = None,
    shadow_sampled: bool | None = None,
) -> None:
    """Set result attributes on a span from `request_span`. No-op if span is None.

    ``gen_ai.usage.input_tokens`` is the prompt actually sent upstream (the
    compressed count); response/output tokens aren't known at this layer, so
    ``gen_ai.usage.output_tokens`` is deliberately never set — backends treat
    it as generated tokens and a prompt count there would corrupt cost math.
    The original/compressed pair lives in the ``distil.*`` namespace.

    Also records the OTel counters. Metrics are recorded here — before the
    ``span is None`` guard — because tracing and metrics are independently
    optional: a fleet may export metrics with sampling-off tracing, and the
    counters must not silently depend on a tracer being installed. Recording at
    this one point also means every call site is covered by construction."""
    record_compression(original_tokens, compressed_tokens)
    if span is None:
        return
    try:
        attrs: dict[str, Any] = {
            "gen_ai.usage.input_tokens": compressed_tokens,
            "distil.tokens.original": original_tokens,
            "distil.tokens.compressed": compressed_tokens,
            "distil.compression.ratio": compression_ratio,
            "distil.compression.applied": compressed,
            "distil.shadow.sampled": shadow_sampled,
        }
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 — observability must never break the request path
        pass
