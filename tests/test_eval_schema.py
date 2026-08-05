"""The published JSON Schema must describe what we actually emit.

A schema that drifts from the payload is worse than none: a consumer validates
against it, passes, and then mis-reads the document. So this validates a REAL
`distil fidelity --json` record against the committed schema on every run.

The validator is written here rather than pulled in as a dependency — distil ships
with zero runtime deps and the test suite keeps that discipline. It covers the
subset of draft 2020-12 the schema actually uses.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from typing import Any

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "schemas" / "eval-record.schema.json"


def _validate(node: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Minimal draft-2020-12 subset: type, required, properties, items, pattern,
    additionalProperties, minimum. Returns a list of human-readable failures."""
    errs: list[str] = []
    types = schema.get("type")
    if types is not None:
        want = types if isinstance(types, list) else [types]
        ok = any(_is_type(node, t) for t in want)
        if not ok:
            return [f"{path}: expected {want}, got {type(node).__name__}"]

    if isinstance(node, dict):
        for key in schema.get("required", []):
            if key not in node:
                errs.append(f"{path}.{key}: required but missing")
        props = schema.get("properties", {})
        for key, value in node.items():
            if key in props:
                errs += _validate(value, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{key}: additional property not permitted")
    elif isinstance(node, list):
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(node):
                errs += _validate(item, item_schema, f"{path}[{i}]")
    elif isinstance(node, str):
        pattern = schema.get("pattern")
        if pattern and not re.match(pattern, node):
            errs.append(f"{path}: {node!r} does not match {pattern}")
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        minimum = schema.get("minimum")
        if minimum is not None and node < minimum:
            errs.append(f"{path}: {node} < minimum {minimum}")
    return errs


def _is_type(node: Any, t: str) -> bool:
    return {
        "object": isinstance(node, dict),
        "array": isinstance(node, list),
        "string": isinstance(node, str),
        "boolean": isinstance(node, bool),
        "integer": isinstance(node, int) and not isinstance(node, bool),
        "number": isinstance(node, (int, float)) and not isinstance(node, bool),
        "null": node is None,
    }.get(t, False)


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _emit(*args: str) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, "-m", "distil.cli", "fidelity", "--json", *args],
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


class TestSchemaFile:
    def test_is_valid_json_and_identifies_itself(self) -> None:
        s = _schema()
        assert s["$schema"].startswith("https://json-schema.org/draft/2020-12")
        assert s["title"] == "distil eval record"

    def test_pins_the_major_version(self) -> None:
        from distil.evalrecord import SCHEMA

        pattern = _schema()["properties"]["schema"]["pattern"]
        assert re.match(pattern, SCHEMA), "the code's SCHEMA constant must satisfy its own schema"


class TestRealRecordValidates:
    def test_passing_run_validates(self) -> None:
        errs = _validate(_emit("--max-silent", "15"), _schema())
        assert errs == [], "emitted record does not match the published schema:\n" + "\n".join(errs)

    def test_failing_run_validates_too(self) -> None:
        """A failing record is still a record — consumers read it precisely then."""
        errs = _validate(_emit("--max-silent", "0"), _schema())
        assert errs == [], "\n".join(errs)

    def test_no_gates_run_validates(self) -> None:
        errs = _validate(_emit(), _schema())
        assert errs == [], "\n".join(errs)


class TestValidatorActuallyRejects:
    """A validator that never fails would make the tests above meaningless."""

    def test_missing_required_field_is_caught(self) -> None:
        doc = _emit("--max-silent", "15")
        del doc["grader"]
        assert any("grader" in e for e in _validate(doc, _schema()))

    def test_unscheme_fingerprint_is_caught(self) -> None:
        """The pattern is scheme-prefixed, deliberately wider than sha256 so a live
        traffic-window descriptor fits. It still rejects an unqualified blob."""
        doc = _emit("--max-silent", "15")
        doc["dataset"]["fingerprint"] = "just-a-bare-string"
        assert any("fingerprint" in e for e in _validate(doc, _schema()))

    def test_live_traffic_descriptor_is_accepted(self) -> None:
        """Live traffic has no content hash and must never grow one — the window
        descriptor plays the same role with different evidence."""
        doc = _emit("--max-silent", "15")
        doc["dataset"]["fingerprint"] = "sig3:n=797+aa405"
        assert _validate(doc, _schema()) == []

    def test_unknown_top_level_key_is_caught(self) -> None:
        doc = _emit("--max-silent", "15")
        doc["surprise"] = 1
        assert any("surprise" in e for e in _validate(doc, _schema()))

    def test_wrong_type_is_caught(self) -> None:
        doc = _emit("--max-silent", "15")
        doc["passed"] = "yes"
        assert any("passed" in e for e in _validate(doc, _schema()))

    def test_metrics_stay_open_for_new_probes(self) -> None:
        """Adding a probe must not break a consumer, so `metrics` is deliberately open."""
        doc = _emit("--max-silent", "15")
        doc["metrics"]["a_brand_new_probe"] = {"x": 1}
        assert _validate(doc, _schema()) == []


class TestShadowRecordValidates:
    """The live path emits the same envelope, so a live result is as attributable
    as an offline one."""

    def _shadow(self, *args: str) -> dict[str, Any]:
        out = subprocess.run(
            [sys.executable, "-m", "distil.cli", "shadow-stats", "--record", *args],
            capture_output=True,
            text=True,
        )
        return json.loads(out.stdout)

    def test_shadow_record_matches_the_schema(self) -> None:
        errs = _validate(self._shadow(), _schema())
        assert errs == [], "\n".join(errs)

    def test_live_grader_is_not_reported_as_synthetic(self) -> None:
        assert self._shadow()["grader"]["kind"] == "live-model-ab"

    def test_traffic_window_is_the_fingerprint(self) -> None:
        d = self._shadow()["dataset"]
        assert d["name"] == "live-traffic"
        assert d["fingerprint"].startswith("sig")
        assert "signature_version" in d, "two runs are only comparable at the same sig version"

    def test_flat_json_shape_is_unchanged(self) -> None:
        """--json is an existing contract; --record is additive, not a replacement."""
        out = subprocess.run(
            [sys.executable, "-m", "distil.cli", "shadow-stats", "--json"],
            capture_output=True,
            text=True,
        )
        d = json.loads(out.stdout)
        assert "schema" not in d, "--json must keep its original flat shape"
        assert {"samples", "changes", "decision_change_rate"} <= set(d)
