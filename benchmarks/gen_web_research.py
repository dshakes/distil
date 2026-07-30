"""Generate corpus/web-research.json — an agent whose tool results are raw HTML.

Why this trajectory has to exist: the HTML transform ships default-on in the serving
path, but every other corpus trajectory carries logs, JSON, or prose, so `distil bench`
never routed a single byte through `compress/htmlx.py`. The transform was certified by
nothing. This closes that.

Shape follows the other trajectories exactly — a byte-stable system+tools prefix, a
growing settling history, one decision-bearing volatile tool output per turn, and one
causally-inert retrieved block for ablation to prune. The difference is that the tool
output is a real HTML document: chrome the extractor must strip, and an article body
carrying the DECISION the oracle grades. If extraction ever swallows the article, the
decision disappears and the gate fails — which is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are an autonomous web-research agent with a headless browser. You answer "
    "questions about live documentation by fetching pages and reading their content.\n"
    "Operating rules:\n"
    "- Fetch a page before citing it; never answer from memory alone.\n"
    "- Prefer the canonical docs origin over blog mirrors.\n"
    "- A page's navigation, cookie banner and footer are chrome, not evidence.\n"
    "DECISION: when a fetched page answers the question, cite it and stop fetching."
)

TOOLS = (
    "AVAILABLE TOOLS (json schema, abbreviated):\n"
    '- fetch_url(url: string) -> {"status": int, "content_type": string, "body": string}\n'
    '- search(query: string, k: int) -> {"results": [{"title","url","snippet"}]}\n'
    '- cite(url: string, quote: string) -> {"ok": bool}\n'
    "Notes: fetch_url returns the raw response body. For text/html that is the full\n"
    "document including script, style and navigation markup.\n"
    "DECISION: call cite() exactly once, with the URL whose body contained the answer."
)


def page(*, title: str, slug: str, body_paras: list[str], decision: str, links: int) -> str:
    """A realistic documentation page: heavy chrome, an article that carries the answer."""
    nav = "".join(
        f'<li><a href="/docs/{slug}/section-{i}">Section {i} of the manual</a></li>'
        for i in range(links)
    )
    script = (
        "<script>window.__ANALYTICS__={id:'UA-000000',queue:[]};"
        + "".join(f"window.__ANALYTICS__.queue.push({{e:{i},t:'pageview'}});" for i in range(40))
        + "</script>"
    )
    style = (
        "<style>"
        + "".join(
            f".c-{i}{{margin:{i}px;padding:{i}px;color:#3{i % 10}{i % 10}}}" for i in range(40)
        )
        + "</style>"
    )
    paras = "".join(f"<p>{p}</p>" for p in body_paras)
    return (
        '<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="utf-8"><title>{title}</title>{script}{style}'
        "</head><body>"
        '<nav class="site-nav"><ul>' + nav + "</ul></nav>"
        '<div class="cookie-banner"><p>We use cookies to improve your experience.</p>'
        "<button>Accept all</button><button>Reject non-essential</button></div>"
        f"<article><header><h1>{title}</h1></header>"
        f"{paras}"
        # The DECISION marker sits alone on its line, with the closing tag on the next.
        # DeterministicRunner splits each LINE on "DECISION:" and compares the remainder
        # as an exact string; with the document minified onto one line the baseline
        # remainder drags the following markup ("...fetching.</p></article><aside") along
        # while the extracted text does not, so the gate would report a divergence about
        # markup rather than about the decision. Whitespace inside <p> is insignificant
        # in HTML, so this changes nothing a browser or a model would see.
        f"<p>\n{decision}\n</p>"
        "</article>"
        '<aside class="related"><h3>Related</h3><ul>'
        + "".join(f'<li><a href="/r/{i}">Related page {i}</a></li>' for i in range(12))
        + "</ul></aside>"
        "<footer><p>Copyright 2026. All rights reserved. "
        "Terms of service. Privacy policy. Status page. Careers.</p></footer>"
        "</body></html>"
    )


PAGES = [
    {
        "title": "Rate limits",
        "slug": "rate-limits",
        "url": "https://docs.example.com/api/rate-limits",
        "body": [
            "Every API key is subject to a per-minute request ceiling and a per-day "
            "token ceiling. The two are enforced independently.",
            "The default tier allows 4000 requests per minute. Exceeding it returns "
            "HTTP 429 with a Retry-After header measured in seconds.",
            "Burst traffic is smoothed over a 10 second window, so a short spike above "
            "the ceiling does not necessarily produce a 429.",
        ],
        "decision": (
            "DECISION: the documented default is 4000 requests per minute, so cite "
            "https://docs.example.com/api/rate-limits and stop fetching."
        ),
        "links": 18,
    },
    {
        "title": "Retry semantics",
        "slug": "retries",
        "url": "https://docs.example.com/api/retries",
        "body": [
            "Clients should retry idempotent requests on 429 and on 5xx, using "
            "exponential backoff with jitter.",
            "The Retry-After header, when present, overrides the client's own backoff "
            "schedule and must be honoured exactly.",
            "Non-idempotent requests must not be retried automatically without an "
            "idempotency key, or the operation may be applied twice.",
        ],
        "decision": (
            "DECISION: Retry-After overrides client backoff, so cite "
            "https://docs.example.com/api/retries and stop fetching."
        ),
        "links": 22,
    },
    {
        "title": "Pagination",
        "slug": "pagination",
        "url": "https://docs.example.com/api/pagination",
        "body": [
            "List endpoints return a cursor in the next_cursor field. An absent or null "
            "cursor means the final page has been reached.",
            "Page size defaults to 100 items and may be raised to 1000 with the limit "
            "parameter. Larger pages cost proportionally more tokens to read.",
            "Cursors are opaque and expire after 24 hours; a stale cursor returns 400.",
        ],
        "decision": (
            "DECISION: a null next_cursor terminates the walk, so cite "
            "https://docs.example.com/api/pagination and stop fetching."
        ),
        "links": 15,
    },
    {
        "title": "Error codes",
        "slug": "errors",
        "url": "https://docs.example.com/api/errors",
        "body": [
            "Every error response carries a machine-readable type field alongside the "
            "human-readable message. Match on type, never on message text.",
            "The invalid_request_error type indicates a client mistake that will not "
            "succeed on retry. The api_error type indicates a server fault that will.",
            "Error messages are not part of the stable interface and change without "
            "notice between releases.",
        ],
        "decision": (
            "DECISION: match on the type field rather than message text, so cite "
            "https://docs.example.com/api/errors and stop fetching."
        ),
        "links": 20,
    },
]

NOISE = [
    "RETRIEVED (speculative context, similarity=0.62): 'Choosing a Frontend Framework "
    "in 2026' — a survey of rendering strategies, hydration cost and bundle budgets "
    "across the major frameworks. Discusses islands architecture and streaming SSR. "
    "No mention of API limits, retries, pagination or error taxonomies.",
    "RETRIEVED (speculative context, similarity=0.58): 'A Field Guide to Colour "
    "Contrast' — how to pick accessible palettes, with worked examples for text on "
    "tinted backgrounds and a note on APCA versus WCAG 2 contrast maths. Unrelated to "
    "the HTTP question under investigation.",
    "RETRIEVED (speculative context, similarity=0.55): 'Notes on Office Espresso' — "
    "grind size, dose and yield for a shared office machine, with a table of extraction "
    "times. Included by the retriever on a spurious keyword overlap with 'limits'.",
    "RETRIEVED (speculative context, similarity=0.61): 'Migrating a Monolith to a "
    "Modular Monolith' — module boundaries, transactional integrity and the cost of "
    "premature service extraction. Architectural, not operational; nothing about the "
    "API surface being researched.",
]

QUESTIONS = [
    "What is the default per-minute request ceiling?",
    "Should a client's own backoff win over Retry-After?",
    "How do I know I have walked every page?",
    "Should I branch on the error message string?",
]

HISTORY = [
    "assistant: fetched the rate-limits page and read the ceiling from the article body; "
    "the navigation and cookie banner carried no evidence.",
    "assistant: fetched the retries page; Retry-After is authoritative over local backoff "
    "when the header is present.",
    "assistant: fetched the pagination page; termination is signalled by a null cursor "
    "rather than by an empty item list.",
]


def build() -> dict:
    turns = []
    for i in range(4):
        blocks = [
            {
                "id": "system",
                "kind": "system",
                "stability": "stable",
                "decision_relevant": True,
                "text": SYSTEM,
            },
            {
                "id": "tools",
                "kind": "tools",
                "stability": "stable",
                "decision_relevant": True,
                "text": TOOLS,
            },
        ]
        # settling history must precede every volatile block (cacheable prefix)
        for h in range(i):
            blocks.append(
                {
                    "id": f"hist-{h}",
                    "kind": "history",
                    "stability": "settling",
                    "decision_relevant": False,
                    "text": HISTORY[h],
                }
            )
        p = PAGES[i]
        html = page(
            title=p["title"],
            slug=p["slug"],
            body_paras=p["body"],
            decision=p["decision"],
            links=p["links"],
        )
        blocks.append(
            {
                "id": f"obs-{i}",
                "kind": "tool_output",
                "stability": "volatile",
                "decision_relevant": True,
                "text": f'fetch_url("{p["url"]}") -> 200 text/html\n{html}',
            }
        )
        blocks.append(
            {
                "id": f"doc-{i}",
                "kind": "retrieved",
                "stability": "volatile",
                "decision_relevant": False,
                "text": NOISE[i],
            }
        )
        blocks.append(
            {
                "id": f"user-{i}",
                "kind": "user",
                "stability": "volatile",
                "decision_relevant": False,
                "text": QUESTIONS[i],
            }
        )
        turns.append({"index": i, "blocks": blocks})

    return {
        "id": "web-research",
        "model": MODEL,
        "_note": (
            "A 4-turn headless web-research agent whose tool results are raw HTML "
            "documents, not logs or JSON. It exists because the HTML transform "
            "(compress/htmlx.py) ships default-on in the serving path while every other "
            "corpus trajectory carries content it never touches — so `distil bench` "
            "certified the transform against nothing. Each page wraps its answer in an "
            "<article> and surrounds it with the chrome extraction is meant to drop "
            "(script, style, nav, cookie banner, aside, footer). The DECISION the oracle "
            "grades lives in the article body, so if extraction ever swallows the "
            "article the decision vanishes and the gate fails."
        ),
        "turns": turns,
    }


if __name__ == "__main__":
    out = Path("corpus/web-research.json")
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
