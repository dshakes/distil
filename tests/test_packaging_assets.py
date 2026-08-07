"""Shipped ops assets must stay consistent with the code.

A Grafana panel or Prometheus alert referencing a metric distil does not emit is
worse than no panel at all: it renders empty, or the alert never fires, and both
read as "the system is fine". These tests fail at authoring time instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "packaging" / "grafana" / "distil-gateway-dashboard.json"
CHART = ROOT / "packaging" / "helm" / "distil-gateway"
RULES = CHART / "templates" / "prometheusrule.yaml"


def _known_metrics() -> set[str]:
    from distil.metrics import _SPEC

    return {name for name, _kind, _help in _SPEC}


def test_dashboard_is_valid_json_with_panels():
    d = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    assert d["uid"] == "distil-gateway"
    assert d["panels"], "a dashboard with no panels is not a dashboard"


def test_every_dashboard_metric_is_actually_emitted():
    d = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    exprs = [t["expr"] for p in d["panels"] for t in p.get("targets", [])]
    used = {m for e in exprs for m in re.findall(r"distil_[a-z_]+", e)}
    assert used, "no metrics referenced — the dashboard would be empty"
    missing = used - _known_metrics()
    assert not missing, f"dashboard references metrics distil does not emit: {sorted(missing)}"


def test_every_alert_rule_metric_is_actually_emitted():
    text = RULES.read_text(encoding="utf-8")
    used = {m for m in re.findall(r"distil_[a-z_]+", text)}
    missing = used - _known_metrics()
    assert not missing, f"alert rules reference metrics distil does not emit: {sorted(missing)}"


def test_helm_chart_has_the_files_helm_requires():
    assert (CHART / "Chart.yaml").is_file()
    assert (CHART / "values.yaml").is_file()
    assert (CHART / "templates" / "deployment.yaml").is_file()


@pytest.mark.parametrize(
    "template",
    sorted(p.name for p in (CHART / "templates").glob("*.yaml")),
)
def test_templates_reference_only_defined_helpers(template):
    """Every `include` must name a helper that _helpers.tpl actually defines."""
    helpers = (CHART / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    defined = set(re.findall(r'define\s+"([^"]+)"', helpers))
    body = (CHART / "templates" / template).read_text(encoding="utf-8")
    used = set(re.findall(r'include\s+"([^"]+)"', body))
    missing = used - defined
    assert not missing, f"{template} includes undefined helper(s): {sorted(missing)}"


def test_chart_defaults_are_secure():
    """The chart must not ship a permissive default that a reviewer has to catch."""
    v = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "requireKeys: true" in v, "an in-cluster gateway must require credentials"
    assert "trustTenantHeader: false" in v, (
        "trusting a client tenant header lets any client bill another"
    )
    assert "runAsNonRoot: true" in v
    assert "readOnlyRootFilesystem: true" in v
