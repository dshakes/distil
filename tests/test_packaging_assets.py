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


# --- README integrity ---------------------------------------------------------
# A broken link in the README is a silent adoption tax: the reader assumes the
# feature does not exist. These are cheap to check and impossible to remember.

README = ROOT / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_relative_links_all_resolve():
    """Every relative link/image target must exist on disk."""
    text = _readme()
    targets = re.findall(r"\]\((?!https?:|#|mailto:)([^)\s]+)\)", text)
    targets += re.findall(r'src="(?!https?:)([^"]+)"', text)
    missing = sorted({t for t in targets if not (ROOT / t.split("#")[0]).exists()})
    assert not missing, f"README links to files that do not exist: {missing}"


def test_readme_internal_anchors_all_exist():
    """Every #anchor must match a heading or an explicit id."""
    text = _readme()

    slugs = set()
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            title = re.sub(r"<[^>]+>", "", m.group(1))
            # GitHub's slug algorithm: lowercase, drop punctuation (emoji
            # included), then spaces -> hyphens. It does NOT strip first, so an
            # emoji heading keeps a leading hyphen: "## 🚀 Use it now" -> "-use-it-now".
            slug = re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "-")
            slugs.add(slug)
            slugs.add(slug.strip("-"))
    slugs |= set(re.findall(r'id="([^"]+)"', text))

    used = {a for a in re.findall(r"\]\(#([^)]+)\)", text)}
    missing = sorted(used - slugs)
    assert not missing, f"README anchors with no matching heading: {missing}"


def test_readme_leads_with_capability_not_statistics():
    """Positioning guard: 'What it does' must appear before the proof section.

    The audit found the README opened with a non-inferiority claim and a
    confidence interval — an answer to a question the reader has not asked yet.
    Capability first, proof below the fold.
    """
    text = _readme()
    assert "## What it does" in text
    assert text.index("## What it does") < text.index("Why trust it"), (
        "the proof section must sit below the capability section"
    )


# --- claims re-audit ----------------------------------------------------------
# GA_READINESS lists "claims re-audit at launch commit" as a launch gate, and the
# last full audit was 1.11.0. A one-time human read goes stale the moment code
# moves, so the checkable claims are pinned here instead: the README cannot
# advertise an agent, an integration, or an API that does not exist, and it
# cannot omit one that does.


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_every_advertised_agent_has_a_real_preset():
    from distil.onboard import AGENT_PRESETS

    text = _readme_text()
    line = next(ln for ln in text.splitlines() if ln.startswith("- **Wrap your agent**"))
    advertised = set(re.findall(r"`(?:distil wrap -- )?([a-z][a-z0-9-]*)`", line))
    unreal = advertised - set(AGENT_PRESETS)
    assert not unreal, f"README advertises agents with no preset: {sorted(unreal)}"


def test_every_real_preset_is_advertised():
    """The reverse: shipping an agent nobody is told about wastes the work."""
    from distil.onboard import AGENT_PRESETS

    text = _readme_text()
    missing = [a for a in AGENT_PRESETS if f"`{a}`" not in text]
    assert not missing, f"presets exist but the README never mentions them: {missing}"


def test_every_advertised_integration_module_exists():
    import importlib

    text = _readme_text()
    line = next(ln for ln in text.splitlines() if ln.startswith("- **Framework hooks**"))
    named = re.findall(r"\b(LangChain|LangGraph|LiteLLM|Agno|Strands)\b", line)
    assert named, "the framework-hooks line names no framework"
    for name in named:
        mod = f"distil.integrations.{name.lower()}"
        importlib.import_module(mod)  # raises if the claim is false


def test_advertised_public_api_names_are_importable():
    """The README's headline code sample must actually run."""
    import distil

    text = _readme_text()
    for name in re.findall(r"from distil import ([a-z_, ]+)", text):
        for symbol in (n.strip() for n in name.split(",")):
            assert hasattr(distil, symbol), f"README imports `{symbol}`, which does not exist"


def test_no_doc_promises_a_proxy_endpoint_that_does_not_exist():
    """A documented endpoint nobody implemented is a support ticket in waiting."""
    import distil.gateway as gw

    served = set(re.findall(r'self\.path == "(/distil/[a-z]+)"', inspect_source(gw)))
    served |= {"/v1/messages", "/v1/chat/completions", "/v1/responses"}
    docs = [README] + sorted((ROOT / "docs").glob("*.md"))
    claimed = set()
    for d in docs:
        text = d.read_text(encoding="utf-8")
        # Only CODE SPANS count. A bare match also catches website paths like
        # `dshakes.github.io/distil/architecture.html` and
        # `github.com/dshakes/distil/actions`, which are pages, not endpoints.
        claimed |= set(re.findall(r"`(/distil/[a-z]+)`", text))
    unreal = claimed - served
    assert not unreal, f"docs reference proxy endpoints that do not exist: {sorted(unreal)}"


def inspect_source(mod) -> str:
    import inspect as _inspect

    return _inspect.getsource(mod)


# --- llms.txt and examples ----------------------------------------------------
# llms.txt is how AI agents discover what distil can do, and it had gone stale
# (still describing "framework hooks for LiteLLM/LangChain/LangGraph" long after
# Agno, Strands, the library API and the TS port shipped). Same failure as the
# README claims: a hand-maintained inventory drifts silently.

LLMS_TXT = ROOT / "docs" / "llms.txt"
EXAMPLES = ROOT / "examples"


def test_llms_txt_lists_every_framework_integration():
    text = LLMS_TXT.read_text(encoding="utf-8")
    modules = {
        p.stem for p in (ROOT / "distil" / "integrations").glob("*.py") if p.stem != "__init__"
    }
    missing = [m for m in modules if m.lower() not in text.lower()]
    assert not missing, f"llms.txt omits shipped integrations: {sorted(missing)}"


def test_llms_txt_lists_every_wrap_preset():
    from distil.onboard import AGENT_PRESETS

    text = LLMS_TXT.read_text(encoding="utf-8").lower()
    missing = [a for a in AGENT_PRESETS if a not in text]
    assert not missing, f"llms.txt omits wrap presets: {sorted(missing)}"


def test_llms_txt_states_the_library_tier_boundary():
    """The one thing an integrator must not get wrong.

    In-process libraries are lossless-only; the digest tier comes from the proxy.
    An agent reading llms.txt and wiring the library expecting digest-level
    savings would conclude distil under-delivers.
    """
    text = LLMS_TXT.read_text(encoding="utf-8").lower()
    assert "lossless" in text and "digest" in text
    assert "certificate" in text


def test_llms_txt_only_links_site_pages_that_exist():
    text = LLMS_TXT.read_text(encoding="utf-8")
    pages = set(re.findall(r"https://dshakes\.github\.io/distil/([a-z0-9-]+\.html)", text))
    missing = sorted(p for p in pages if not (ROOT / "docs" / p).is_file())
    assert not missing, f"llms.txt links site pages that do not exist: {missing}"


def test_every_example_referenced_in_docs_exists():
    """A README pointing at an example nobody wrote is a dead end at the worst moment."""
    referenced = set()
    for doc in (ROOT / "README.md", EXAMPLES / "README.md"):
        referenced |= set(
            re.findall(r"examples/([A-Za-z0-9_.-]+\.(?:py|ts))", doc.read_text("utf-8"))
        )
    missing = sorted(f for f in referenced if not (EXAMPLES / f).is_file())
    assert not missing, f"docs reference examples that do not exist: {missing}"


def test_new_library_examples_are_indexed():
    """Writing an example nobody is pointed to wastes it."""
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for name in ("python_library.py", "js_library.ts", "js_ai_sdk_middleware.ts"):
        assert name in index, f"{name} exists but examples/README.md never mentions it"


def test_site_search_index_covers_every_page():
    """The index is generated; a stale one silently hides new documentation."""
    index = json.loads((ROOT / "docs" / "search-index.json").read_text(encoding="utf-8"))
    # Honour the builder's own exclusions rather than restating them here — a
    # second copy of the rule is a second thing to keep in sync.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bsi", ROOT / "scripts" / "build_search_index.py"
    )
    bsi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsi)

    indexed = {e["u"] for e in index}
    pages = {p.name for p in (ROOT / "docs").glob("*.html")} - set(bsi.SKIP)
    missing = sorted(pages - indexed)
    assert not missing, (
        f"search index is stale — run scripts/build_search_index.py. Missing: {missing}"
    )
