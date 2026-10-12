"""Every ``DISTIL_*`` environment variable the code reads, in one registry.

``kind`` is one of:

* ``user``     a knob a user may set; documented in docs/cli.html (generated from here).
* ``internal`` set by distil itself for its own child processes; never set by hand.
* ``test``     a test hook; read ONLY when ``DISTIL_TESTING=1`` (see ``distil._testing``).

tests/test_envvars.py fails if code reads a variable that is not listed, or if the
docs table drifts from this registry. Regenerate the table: ``python -m distil.envvars --write``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# (name, kind, default, what it does)
VARS: tuple[tuple[str, str, str, str], ...] = (
    (
        "DISTIL_HOME",
        "user",
        "~/.distil",
        "State directory: ledger, restore store, config. Isolates a test or CI run from your real state.",
    ),
    (
        "DISTIL_NO_TELEMETRY",
        "user",
        "unset",
        "Set to 1 to disable the opt-in census regardless of a stored opt-in (as does DO_NOT_TRACK=1).",
    ),
    (
        "DISTIL_NO_UPDATE_CHECK",
        "user",
        "unset",
        "Set to 1 to skip the once-a-day new-version check.",
    ),
    (
        "DISTIL_NO_LEDGER",
        "user",
        "unset",
        "Set to 1 to stop the proof ledger being printed on exit.",
    ),
    (
        "DISTIL_NO_DRIFT_GUARD",
        "user",
        "unset",
        "Set to 1 to opt out of the drift guard that holds compression after evidence of drift.",
    ),
    (
        "DISTIL_NO_ENCRYPT_AT_REST",
        "user",
        "unset",
        "Set to 1 to store the restore data unencrypted (strict-FS setups).",
    ),
    (
        "DISTIL_CENSUS_ENDPOINT",
        "user",
        "distil-census.vercel.app/v1/ping",
        "Where the opt-in, content-free daily census is POSTed (self-hosted ingest). "
        "https:// only, or http://localhost / 127.0.0.1; any other value is ignored with a note.",
    ),
    (
        "DISTIL_BEAT_ENDPOINT",
        "user",
        "distil-census.vercel.app/v1/beat",
        "Where the opt-in, content-free savings heartbeat is POSTed. Same rules as "
        "DISTIL_CENSUS_ENDPOINT.",
    ),
    (
        "DISTIL_RESTORE_CAP",
        "user",
        "5000",
        "Max entries in the restore store (LRU). Older digests then expand to a placeholder.",
    ),
    ("DISTIL_RESTORE_TTL_DAYS", "user", "14", "Days a restore-store entry is kept."),
    (
        "DISTIL_COLD_POINT",
        "user",
        "1",
        "0 turns off cold-point recompression (ADR 0014); same as --no-cold-point.",
    ),
    (
        "DISTIL_REFETCH_VERBATIM",
        "user",
        "1",
        "0 turns off re-fetch verbatim (ADR 0025). Read per request.",
    ),
    (
        "DISTIL_SH_OFF",
        "user",
        "unset",
        "Set to 1 and distil sh runs every command untouched; the Claude Code shell rewrite leaves commands as written (ADR 0026).",
    ),
    ("DISTIL_SHADOW_PAIRED", "user", "1", "0 restores the cheaper unpaired shadow estimator."),
    (
        "DISTIL_HOT_SWAP",
        "user",
        "1",
        "0 keeps the historical in-thread proxy instead of the hot-swappable worker (POSIX).",
    ),
    (
        "DISTIL_SUBSCRIPTION",
        "user",
        "auto",
        "1/0 forces subscription (tokens-only) or metered ($) reporting; auto detects an OAuth login.",
    ),
    (
        "DISTIL_HOLDOUT_RATE",
        "user",
        "unset",
        "Holdout share for new sessions, 0-1 (0 off); beats ab.json.",
    ),
    (
        "DISTIL_AB_COST_CAP",
        "user",
        "built in",
        "Overrides the per-task cost cap in `distil ab` (changing it changes the estimand).",
    ),
    (
        "DISTIL_GATEWAY_TOKEN",
        "user",
        "unset",
        "Admin token for `distil gateway`; required on a non-loopback bind.",
    ),
    ("DISTIL_OIDC_ISSUER", "user", "unset", "Gateway OIDC: expected token issuer."),
    ("DISTIL_OIDC_AUDIENCE", "user", "unset", "Gateway OIDC: expected audience."),
    ("DISTIL_OIDC_HS256_SECRET", "user", "unset", "Gateway OIDC: HS256 shared secret."),
    ("DISTIL_OIDC_PUBLIC_KEY", "user", "unset", "Gateway OIDC: PEM public key for RS/ES tokens."),
    ("DISTIL_OIDC_ROLE_CLAIM", "user", "role", "Gateway OIDC: claim carrying the role."),
    ("DISTIL_OIDC_TENANT_CLAIM", "user", "tenant", "Gateway OIDC: claim carrying the tenant."),
    (
        "DISTIL_CA_BUNDLE",
        "user",
        "unset",
        "CA bundle for upstream TLS (also honours REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE, SSL_CERT_FILE).",
    ),
    ("DISTIL_UPSTREAM_TIMEOUT", "user", "600", "Seconds to wait on the provider."),
    ("DISTIL_CLIENT_TIMEOUT", "user", "600", "Seconds to wait on the calling client."),
    ("DISTIL_MCP_TIMEOUT", "user", "600", "Seconds to wait on a proxied MCP server."),
    (
        "DISTIL_WORKER_READY_TIMEOUT",
        "user",
        "30",
        "Seconds the hot-swap supervisor waits for a replacement worker before rolling back.",
    ),
    (
        "DISTIL_DRAIN_BUDGET_S",
        "user",
        "300",
        "Seconds a retiring worker may drain in-flight requests during a hot swap.",
    ),
    (
        "DISTIL_DEBUG",
        "user",
        "unset",
        "Set to 1 to write swallowed (fail-open) exceptions to stderr.",
    ),
    (
        "DISTIL_LOG_LEVEL",
        "user",
        "unset",
        "DEBUG, INFO, ... for the same log; DISTIL_DEBUG=1 is DEBUG.",
    ),
    (
        "DISTIL_STATUSLINE",
        "user",
        "full",
        "minimal, lite or compact selects the two-fact status line.",
    ),
    (
        "DISTIL_IGNORE_SETTINGS_PRECEDENCE",
        "user",
        "unset",
        "Set to skip the settings-precedence check in `distil wrap`.",
    ),
    ("DISTIL_VISION", "user", "auto", "1/0 force-enables or hard-disables image compression."),
    (
        "DISTIL_VISION_DOWNSCALE",
        "user",
        "auto",
        "1/0 force-enables or disables image downscaling (needs a passing local certify).",
    ),
    (
        "DISTIL_LITELLM_DIGEST",
        "user",
        "unset",
        "Set to 1 to opt the LiteLLM hook into Tier-1 digests.",
    ),
    (
        "DISTIL_CORPUS",
        "user",
        "packaged",
        "Corpus directory for bench, savings and the research commands.",
    ),
    (
        "DISTIL_SESSION",
        "internal",
        "-",
        "Session id; set by `distil wrap` for the agent it launches.",
    ),
    (
        "DISTIL_SURFACE",
        "internal",
        "-",
        "Which surface launched the proxy (wrap, proxy, gateway); set by the CLI.",
    ),
    ("DISTIL_MCP_SESSION", "internal", "-", "Session id passed to a proxied MCP server."),
    (
        "DISTIL_WORKER_CONFIG",
        "internal",
        "-",
        "Serialized worker config handed to the hot-swap worker.",
    ),
    ("DISTIL_WORKER_FD", "internal", "-", "Inherited listening-socket fd for the hot-swap worker."),
    (
        "DISTIL_E7_GATE_RECENT",
        "internal",
        "-",
        "Printed by `distil research online` for the harness; never read by distil.",
    ),
    (
        "DISTIL_TESTING",
        "test",
        "unset",
        "Set to 1 to enable the test hooks below. Never set in production.",
    ),
    (
        "DISTIL_HOTSWAP_TEST_FAIL_READY",
        "test",
        "-",
        "Hot-swap worker exits before READY if this file exists.",
    ),
)

BEGIN = "<!-- envvars:begin (generated by python -m distil.envvars --write) -->"
END = "<!-- envvars:end -->"


def render_table() -> str:
    rows = "\n".join(
        f'        <tr><th scope="row"><code>{n}</code></th><td>{d}</td><td>{h}</td></tr>'
        for n, k, d, h in VARS
        if k == "user"
    )
    return (
        f'{BEGIN}\n    <div class="table-scroll" tabindex="0" role="region" '
        f'aria-label="Environment variables">\n    <table>\n      <thead><tr><th scope="col">Variable</th>'
        f'<th scope="col">Default</th><th scope="col">Effect</th></tr></thead>\n'
        f"      <tbody>\n{rows}\n      </tbody>\n    </table>\n    </div>\n    {END}"
    )


def _splice(doc: str) -> str:
    a, b = doc.index(BEGIN), doc.index(END) + len(END)
    return doc[:a] + render_table() + doc[b:]


if __name__ == "__main__":
    p = Path(__file__).resolve().parent.parent / "docs" / "cli.html"
    if "--write" in sys.argv:
        p.write_text(_splice(p.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        print(render_table())
