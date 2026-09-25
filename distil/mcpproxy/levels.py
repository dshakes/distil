"""The compression levels, as pure functions — no I/O except the restore-store write in R.

Every level is an explicit, named mode. Definitions (what the model is told it can
call) and results (what a call returned) are compressed independently:

``L0`` lossless
    Schema canonicalisation that provably does not change what the model can call:
    only JSON-Schema *annotation* keywords that carry no validation meaning are
    removed (``$comment``; a ``title`` derivable from its property/tool name; an empty
    ``description``), ``$schema`` is dropped only where the draft-07 → 2020-12 dialect
    change is a no-op for that schema, and description whitespace is normalised.
    Every validation keyword, every ``enum``/``const``/``default``/``examples`` value
    and every non-empty description's words are kept.
``L1`` summary
    L0, plus every description cut to an *extractive* summary: its first sentence and
    up to two sentences that state a constraint (must / never / only / default …),
    verbatim. Never generated, never longer than the original. The full description
    is one ``<server>_get_tool_schema`` call away.
``L2`` lazy
    One stable index per server (``name(args): first sentence``) in a single tool's
    description. Fetching a tool's schema *unlocks* the real tool — original name,
    real ``inputSchema`` — for the rest of the session (the client is told via
    ``notifications/tools/list_changed``). ``<server>_invoke_tool`` stays as the
    fallback for clients that never refresh their tool list. The unlocked set only
    grows and is appended at the END of the list, so a cached prefix survives.
``L3`` adaptive
    L2, plus the tools this server has actually been used for (learned locally, names
    and counts only) pinned fully expanded from the first turn. The pin set is fixed
    for a session, so it never moves the list mid-session.
``R`` results (orthogonal)
    Large text results are digested with distil's recoverable digest (columnar fold
    for JSON record arrays, tier-1 digest otherwise), the original persisted to the
    shared restore store and recoverable through ``<server>_expand``. Rejected when
    not smaller. Never touches errors, non-text content, results carrying
    ``structuredContent`` (the client parses those), or tools whose output an agent
    must quote back byte-exact (file reads).
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..compress import structured
from ..compress.provenance import EXACT_QUOTE_TOOLS
from ..compress.tier1 import _handle, digest
from ..tokenizer import DEFAULT as _tokenizer

LEVELS = ("L0", "L1", "L2", "L3")
LEVEL_NAMES = {"L0": "lossless", "L1": "summary", "L2": "lazy", "L3": "adaptive"}
#: Until the pre-registered live run certifies a more aggressive level, the default is
#: the only one that is equivalent by construction. See docs/adr/0017-mcp-compressor.md.
DEFAULT_LEVEL = "L0"
#: R (result digests) is OFF until its own certificate is issued — ADR 0017, review 2026-09-25.
DEFAULT_RESULTS = False

# ---------------------------------------------------------------------------
# Token accounting — what a client actually shows the model
# ---------------------------------------------------------------------------


def model_view(tool: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a tool definition a client puts in front of the model.

    ``title``, ``annotations``, ``outputSchema`` and ``execution`` are client-side
    metadata (display names, approval hints, result validation); the model-facing
    projection every major client sends is name + description + input schema.
    """
    out: dict[str, Any] = {"name": tool.get("name", "")}
    if "description" in tool:
        out["description"] = tool["description"]
    out["inputSchema"] = tool.get("inputSchema", {})
    return out


def tokens(obj: Any) -> int:
    """Token size of *obj* as compact JSON (strings are counted as-is)."""
    text = (
        obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    )
    return _tokenizer.count(text)


def definition_tokens(tool: Mapping[str, Any]) -> int:
    return tokens(model_view(tool))


# ---------------------------------------------------------------------------
# L0 — lossless schema canonicalisation
# ---------------------------------------------------------------------------

# JSON-Schema positions that hold subschemas. Only dict nodes reached through these are
# treated as schemas, so a *property named* "title" or "$comment" is never mistaken for
# the keyword of the same name.
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_ONES = (
    "items",
    "additionalProperties",
    "additionalItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
)

# The two dialects MCP servers actually emit. MCP's current revision reads a schema with
# no ``$schema`` as 2020-12; draft-07 (zod-to-json-schema's default) differs from it only
# in the keywords ``_dialect_sensitive`` looks for, so without them dropping it is a no-op.
_DROPPABLE_DIALECTS = frozenset(
    {
        "http://json-schema.org/draft-07/schema#",
        "http://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft/2020-12/schema",
    }
)

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _norm(s: str) -> str:
    return _NON_ALNUM.sub("", s.lower())


def _ws(text: str) -> str:
    """Trailing spaces off every line, at most one blank line in a row, ends trimmed."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _dialect_sensitive(node: Any) -> bool:
    """Does *node* use a keyword whose meaning differs between draft-07 and 2020-12?"""
    if isinstance(node, dict):
        if isinstance(node.get("items"), list) or "additionalItems" in node:
            return True
        if "dependencies" in node or "$recursiveRef" in node or "$dynamicRef" in node:
            return True
        if "$ref" in node and len(node) > 1:  # draft-07 ignores $ref's siblings
            return True
        for k in ("exclusiveMinimum", "exclusiveMaximum"):
            if isinstance(node.get(k), bool):  # draft-04 form
                return True
        return any(_dialect_sensitive(v) for v in node.values())
    if isinstance(node, list):
        return any(_dialect_sensitive(v) for v in node)
    return False


DescFn = Callable[[str], str]


def canonical_schema(
    node: Any, *, name_hint: str | None = None, root: bool = True, desc: DescFn = _ws
) -> Any:
    """Return a canonical copy of a JSON-Schema node (input untouched).

    Removes only: ``$comment``; ``title`` when it merely restates the property / tool
    name it sits under; ``description`` when empty after whitespace normalisation;
    ``$schema`` at the root when dropping it cannot change the dialect's meaning.
    ``desc`` rewrites description strings (whitespace-only for L0).
    """
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    drop_dialect = (
        root and node.get("$schema") in _DROPPABLE_DIALECTS and not _dialect_sensitive(node)
    )
    out: dict[str, Any] = {}
    for key, val in node.items():
        if key == "$comment":
            continue
        if key == "$schema" and drop_dialect:
            continue
        if key == "title" and isinstance(val, str) and name_hint and _norm(val) == _norm(name_hint):
            continue
        if key == "description" and isinstance(val, str):
            val = desc(val)
            if not val:
                continue
            out[key] = val
            continue
        if key in _SCHEMA_MAPS and isinstance(val, dict):
            out[key] = {
                k: canonical_schema(v, name_hint=k, root=False, desc=desc) for k, v in val.items()
            }
        elif key in _SCHEMA_LISTS and isinstance(val, list):
            out[key] = [canonical_schema(v, root=False, desc=desc) for v in val]
        elif key in _SCHEMA_ONES and isinstance(val, list):  # items: [..] tuple form
            out[key] = [canonical_schema(v, root=False, desc=desc) for v in val]
        elif key in _SCHEMA_ONES and isinstance(val, dict):
            out[key] = canonical_schema(val, root=False, desc=desc)
        else:
            out[key] = copy.deepcopy(val)
    return out


def canonical_tool(tool: Mapping[str, Any], *, desc: DescFn = _ws) -> dict[str, Any]:
    """L0 (or, with ``desc=summarize``, L1) form of one tool definition."""
    name = str(tool.get("name", ""))
    out: dict[str, Any] = {}
    for key, val in tool.items():
        if key == "title" and isinstance(val, str) and _norm(val) == _norm(name):
            continue
        if key == "description" and isinstance(val, str):
            val = desc(val)
            if not val:
                continue
        elif key == "inputSchema" and isinstance(val, dict):
            val = canonical_schema(val, name_hint=name, desc=desc)
        else:
            val = copy.deepcopy(val)
        out[key] = val
    return out


def dropped_paths(before: Any, after: Any, path: str = "") -> list[tuple[str, str]]:
    """``(json-pointer, change)`` for everything *after* removed or rewrote.

    ``change`` is ``removed`` or ``shortened``. Used by the webdash diff view and by the
    L0 equivalence test, which asserts every removal is an annotation keyword.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[tuple[str, str]] = []
        for k, v in before.items():
            p = f"{path}/{k.replace('~', '~0').replace('/', '~1')}"
            if k not in after:
                out.append((p, "removed"))
            else:
                out += dropped_paths(v, after[k], p)
        return out
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [
            x
            for i, (b, a) in enumerate(zip(before, after))
            for x in dropped_paths(b, a, f"{path}/{i}")
        ]
    if before != after:
        return [(path or "/", "shortened")]
    return []


# ---------------------------------------------------------------------------
# L1 — extractive summary
# ---------------------------------------------------------------------------

# Sentence ends, paragraph breaks, and list items all start a new unit.
_SPLIT_RE = re.compile(r"(?<=[^\d\s][.!?])\s+(?=[^\s])|\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")
_CUE_RE = re.compile(
    r"\b(must|never|only|required|requires|cannot|can't|do not|don't|not supported|"
    r"warning|important|deprecated|at most|at least|maximum|minimum|default)\b",
    re.IGNORECASE,
)
#: How many constraint sentences an L1 summary keeps after the first sentence.
L1_MAX_CUES = 2


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT_RE.split(_ws(text)) if s and s.strip()]


def summary_units(text: str, max_cues: int = L1_MAX_CUES) -> list[str]:
    """The sentences an L1 summary keeps, verbatim and in original order."""
    units = sentences(text)
    if len(units) <= 1:
        return units
    cues = [s for s in units[1:] if _CUE_RE.search(s)][:max_cues]
    keep = [units[0], *cues]
    return [s for s in units if s in keep]


def summarize(text: str, max_cues: int = L1_MAX_CUES) -> str:
    """First sentence + up to ``max_cues`` constraint sentences, verbatim, in order.

    Deterministic and extractive: every unit returned occurs in *text*. Returns the
    whitespace-normalised original when summarising would not make it shorter.
    """
    full = _ws(text)
    out = " ".join(summary_units(full, max_cues))
    return out if out and len(out) < len(full) else full


# ---------------------------------------------------------------------------
# L2 / L3 — the lazy surface
# ---------------------------------------------------------------------------

#: L3 pins at most this many tools, and only those used at least ``PIN_MIN_CALLS`` times.
PIN_TOP_K = 8
PIN_MIN_CALLS = 3
_MAX_TOOL_NAME = 64  # the tightest limit among the major clients' tool-name rules


def safe_server(server: str) -> str:
    """Letters, digits and ``-`` only. No ``_`` at all, so in a namespaced name
    ``<server>__<tool>`` the first ``__`` always ends the server part: one server can
    never mint a name that parses as another server's."""
    return re.sub(r"[^A-Za-z0-9-]+", "-", server).strip("-") or "mcp"


#: Separator between server and tool when one proxy fronts several servers.
NAMESPACE_SEP = "__"


def meta_name(server: str, suffix: str, sep: str = "_") -> str:
    base = safe_server(server)[: _MAX_TOOL_NAME - len(suffix) - len(sep)]
    return f"{base}{sep}{suffix}"


def signature(tool: Mapping[str, Any]) -> str:
    """``name(a, b?)`` — required args bare, optional ones with ``?``."""
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    args = [k if k in required else f"{k}?" for k in (props or {})]
    return f"{tool.get('name', '')}({', '.join(args)})"


def index_line(tool: Mapping[str, Any]) -> str:
    first = summarize(str(tool.get("description") or ""), max_cues=0)
    first = first.split("\n", 1)[0]
    return f"{signature(tool)}: {first}" if first else signature(tool)


def choose_pins(usage: Mapping[str, int], names: Iterable[str]) -> frozenset[str]:
    """L3's pin set: the most-used tools (ties by name), learned from local counts."""
    avail = set(names)
    ranked = sorted(
        ((n, c) for n, c in usage.items() if n in avail and c >= PIN_MIN_CALLS),
        key=lambda nc: (-nc[1], nc[0]),
    )
    return frozenset(n for n, _ in ranked[:PIN_TOP_K])


@dataclass
class Surface:
    """What one backend server looks like to the client at one level, for one session."""

    server: str
    tools: list[dict[str, Any]]
    level: str = DEFAULT_LEVEL
    results: bool = DEFAULT_RESULTS
    pinned: frozenset[str] = frozenset()
    prefix: str = ""  # "<server>__" whenever the proxy fronts more than one server
    unlocked: list[str] = field(default_factory=list)
    sep: str = "_"  # meta-tool separator; "__" when namespaced

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown level {self.level!r}; choose one of {', '.join(LEVELS)}")
        self.schema_tool = meta_name(self.server, "get_tool_schema", self.sep)
        self.invoke_tool = meta_name(self.server, "invoke_tool", self.sep)
        self.expand_tool = meta_name(self.server, "expand", self.sep)
        metas = {self.schema_tool, self.invoke_tool, self.expand_tool}
        # A tool whose exposed name would equal one of our meta tools, or a second tool
        # with a name already taken, is SHADOWED: dropped, never routed, and reported
        # (``shadowed``) so the proxy can log it. Ours always win; first listing wins.
        self.by_name: dict[str, dict[str, Any]] = {}
        self.shadowed: list[str] = []
        for t in self.tools:
            if not isinstance(t, dict):
                continue
            n = str(t.get("name"))
            if n in self.by_name or self.prefix + n in metas:
                self.shadowed.append(n)
                continue
            self.by_name[n] = t
        self.unlocked = [n for n in dict.fromkeys(self.unlocked) if n in self.by_name]
        # Reject-if-bigger, applied to the whole surface: a server with a handful of
        # one-line tools gains nothing from an index or a summary, and the meta tools
        # would make its list LARGER. Such a server is served at L0 instead, and
        # ``requested`` keeps what was asked for so the watch view can say so.
        self.requested = self.level
        if self.level != "L0":
            asked = self._list_tokens()
            self.level = "L0"
            if asked < self._list_tokens():
                self.level = self.requested
            else:
                self.unlocked = []

    def _list_tokens(self) -> int:
        return sum(tokens(model_view(t)) for t in self.tools_list())

    @property
    def lazy(self) -> bool:
        return self.level in ("L2", "L3")

    def visible(self, name: str) -> bool:
        return not self.lazy or name in self.pinned or name in self.unlocked

    def definition(self, name: str) -> dict[str, Any]:
        """The compressed definition of backend tool *name*, under its exposed name.

        Lazily surfaced tools get the L0 form: the model asked for this schema, so it
        gets the real one.
        """
        desc = summarize if self.level == "L1" else _ws
        out = canonical_tool(self.by_name[name], desc=desc)
        out["name"] = self.prefix + name
        return out

    def meta_tools(self) -> list[dict[str, Any]]:
        server = self.server
        out: list[dict[str, Any]] = []
        if self.lazy:
            index = "\n".join(index_line(t) for t in self.by_name.values())
            out.append(
                {
                    "name": self.schema_tool,
                    "description": (
                        f"Index of the {len(self.by_name)} tools on the '{server}' MCP server. "
                        "Their full schemas load on demand: call this with a tool_name to get "
                        "one, and that tool is added to your tool list under its real name — "
                        f"then call it directly (or via {self.invoke_tool} if your tool list "
                        "does not update). `?` marks optional arguments.\n" + index
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {"tool_name": {"type": "string"}},
                        "required": ["tool_name"],
                    },
                }
            )
            out.append(
                {
                    "name": self.invoke_tool,
                    "description": (
                        f"Fallback: call a '{server}' tool by name. Use only when the tool you "
                        "need is not in your tool list after fetching its schema."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["tool_name"],
                    },
                }
            )
        elif self.level == "L1":
            out.append(
                {
                    "name": self.schema_tool,
                    "description": (
                        f"The '{server}' tools' descriptions are shortened to their first "
                        "sentence and key constraints. Call this with a tool_name for its full "
                        "description and schema."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {"tool_name": {"type": "string"}},
                        "required": ["tool_name"],
                    },
                }
            )
        if self.results:
            out.append(
                {
                    "name": self.expand_tool,
                    "description": (
                        f"Recover the exact original text behind a `handle=` marker in a "
                        f"'{server}' tool result that was shortened to save context."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {"handle": {"type": "string", "pattern": "^[0-9a-f]{8}$"}},
                        "required": ["handle"],
                    },
                }
            )
        return out

    def tools_list(self) -> list[dict[str, Any]]:
        """Deterministic: the same inputs give byte-identical output, every call.

        Non-lazy levels list the real tools in backend order with the meta tools
        appended, so the real-tool prefix matches the uncompressed list's order. Lazy
        levels list the meta tools, then pins (by name), then unlocked tools in unlock
        order — growth is append-only.
        """
        if not self.lazy:
            return [self.definition(n) for n in self.by_name] + self.meta_tools()
        pins = [self.definition(n) for n in sorted(self.pinned) if n in self.by_name]
        extra = [self.definition(n) for n in self.unlocked if n not in self.pinned]
        return self.meta_tools() + pins + extra

    def unlock(self, name: str) -> bool:
        """Surface *name* for the rest of the session. True iff the list changed."""
        if not self.lazy or name not in self.by_name or self.visible(name):
            return False
        self.unlocked.append(name)
        return True

    def resolve(self, exposed: str) -> tuple[str, str | None]:
        """Map a called name to ``(kind, backend_tool)``.

        kind is ``schema`` | ``invoke`` | ``expand`` | ``tool`` | ``unknown``. A real
        tool is routed even when it was never surfaced (a model that already knows the
        name should not be forced through the index), and auto-unlocked by the caller.
        """
        if exposed == self.schema_tool and self.level != "L0":
            return "schema", None
        if exposed == self.invoke_tool and self.lazy:
            return "invoke", None
        if exposed == self.expand_tool and self.results:
            return "expand", None
        if exposed.startswith(self.prefix):
            name = exposed[len(self.prefix) :]
            if name in self.by_name:
                return "tool", name
        return "unknown", None

    def exposure(self) -> list[tuple[str, str, str | None]]:
        """Every name this surface answers to: ``(exposed, kind, backend_tool)``, ours first.

        Real tools are included whether or not they are currently listed, because a
        lazily hidden tool is still routed when called by name. The proxy builds its
        single name→server routing table from this.
        """
        out: list[tuple[str, str, str | None]] = []
        if self.level != "L0":
            out.append((self.schema_tool, "schema", None))
        if self.lazy:
            out.append((self.invoke_tool, "invoke", None))
        if self.results:
            out.append((self.expand_tool, "expand", None))
        out += [(self.prefix + n, "tool", n) for n in self.by_name]
        return out

    def drop(self, name: str) -> None:
        """Withdraw backend tool *name* (a shadowed name the proxy refused to route)."""
        self.by_name.pop(name, None)
        self.unlocked = [n for n in self.unlocked if n != name]
        self.shadowed.append(name)

    def schema_text(self, name: str) -> str:
        """What ``get_tool_schema`` returns: full description + the real input schema."""
        tool = self.by_name[name]
        schema = canonical_schema(tool.get("inputSchema") or {}, name_hint=name)
        desc = _ws(str(tool.get("description") or ""))
        head = f"{signature(tool)}: {desc}" if desc else signature(tool)
        note = ""
        if self.lazy:
            note = (
                f"\n\n`{self.prefix + name}` is now in your tool list — call it directly. If it "
                f"is not, call {self.invoke_tool} with tool_name={name!r}."
            )
        return f"{head}\n\n{json.dumps(schema, indent=1, ensure_ascii=False)}{note}"

    def suggest(self, name: str) -> list[str]:
        import difflib

        return difflib.get_close_matches(name, list(self.by_name), n=3, cutoff=0.5)


# ---------------------------------------------------------------------------
# R — result compression
# ---------------------------------------------------------------------------

#: Text shorter than this is left alone: a digest marker is not worth it.
RESULT_MIN_TOKENS = 200
#: MCP-server spellings of "the agent will quote this back byte-exact" that the shared
#: EXACT_QUOTE_TOOLS set (named for Claude Code / Codex / common filesystem servers)
#: does not already carry.
MCP_EXACT_QUOTE_TOOLS = frozenset(
    {"read_text_file", "read_multiple_files", "get_file_contents", "read_resource"}
)
_READ_FILE_RE = re.compile(r"read.*file|file.*read|get_file|cat_file")
#: Verbs whose result is content an agent may quote or edit against, as a whole word of
#: the tool name (``view``, ``open_document``, ``get_contents``, ``git_show`` …).
_EXACT_WORD_RE = re.compile(
    r"(?:^|[_\-.])(read|view|open|cat|show|contents?|get_contents|file_contents|head|tail|source|blob)(?:[_\-.]|$)"
)
#: A read-only tool whose description says it returns file/source content.
_CONTENT_DESC_RE = re.compile(
    r"\b(file|files|contents?|source code|document|blob|raw text)\b", re.IGNORECASE
)


def exact_quote(tool: str, tool_def: Mapping[str, Any] | None = None) -> bool:
    """Is *tool*'s output something an agent may need back byte-exact? Conservative.

    True for the shared ``EXACT_QUOTE_TOOLS`` names, the MCP servers' spellings, any
    name containing a read/view/open/cat/show/contents-style word, and any tool whose
    definition says ``readOnlyHint: true`` and describes returning file or source
    content. Erring towards True only ever means a result is left uncompressed.
    """
    low = tool.lower()
    if low in EXACT_QUOTE_TOOLS or low in MCP_EXACT_QUOTE_TOOLS:
        return True
    if _READ_FILE_RE.search(low) or _EXACT_WORD_RE.search(low):
        return True
    if isinstance(tool_def, Mapping):
        ann = tool_def.get("annotations")
        read_only = isinstance(ann, Mapping) and ann.get("readOnlyHint") is True
        if read_only and _CONTENT_DESC_RE.search(str(tool_def.get("description") or "")):
            return True
    return False


@dataclass
class ResultInfo:
    tokens_before: int = 0
    tokens_after: int = 0
    handles: list[str] = field(default_factory=list)
    skipped: str | None = None  # why nothing was attempted


def _compress_text(text: str) -> str | None:
    """A smaller recoverable form of *text*, or None."""
    for folder in (structured.fold, structured.fold_records):
        folded = folder(text)
        if folded is not None:
            return folded
    out, changed = digest(text)
    return out if changed else None


def compress_result(
    tool: str,
    result: Any,
    *,
    record: Callable[[str, str], bool],
    min_tokens: int = RESULT_MIN_TOKENS,
    tool_def: Mapping[str, Any] | None = None,
) -> tuple[Any, ResultInfo]:
    """Digest the large text blocks of a ``tools/call`` result. Input is never mutated.

    ``record(handle, original)`` persists the original; returning False (a genuine
    handle collision) leaves that block verbatim.
    """
    info = ResultInfo()
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        info.skipped = "shape"
        return result, info
    if result.get("isError"):
        info.skipped = "error"
        return result, info
    if result.get("structuredContent") is not None:
        info.skipped = "structured"
        return result, info
    if exact_quote(tool, tool_def):
        info.skipped = "exact-quote"
        return result, info
    new_content: list[Any] = []
    changed = False
    for item in result["content"]:
        text = item.get("text") if isinstance(item, dict) and item.get("type") == "text" else None
        if not isinstance(text, str):
            new_content.append(item)
            continue
        before = _tokenizer.count(text)
        info.tokens_before += before
        small = _compress_text(text) if before >= min_tokens else None
        after = _tokenizer.count(small) if small is not None else before
        if small is None or after >= before or not record(_handle(text), text):
            info.tokens_after += before
            new_content.append(item)
            continue
        info.tokens_after += after
        info.handles.append(_handle(text))
        new_content.append({**item, "text": small})
        changed = True
    if not changed:
        return result, info
    return {**result, "content": new_content}, info
