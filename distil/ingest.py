"""Ingest real provider traffic into Distil trajectories.

Converts captured Anthropic Messages API or OpenAI Chat Completions request
bodies into ``Trajectory`` objects so users can run savings / pruning /
cache-aware simulation on their own traffic — closing the "synthetic corpus"
gap.

Honesty note
------------
Real provider traces carry no ``DECISION:`` markers, so the offline
deterministic certify gate (``DeterministicRunner``) does not apply to
ingested trajectories.  Use ingested trajectories for:

* **savings / pruning / cache simulation** — ``distil.compress.cache_aware.simulate``
* **live certification** — re-run the conversation with ``--runner anthropic``
  and certify against the live model's responses.

``decision_relevant`` is always ``False`` on ingested blocks for this reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from .pricing import DEFAULT_MODEL
from .trajectory import Block, Kind, Stability, Trajectory, Turn


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


def ingest_anthropic_request(body: dict, *, turn_index: int = 0) -> list[Block]:
    """Map one Anthropic Messages request body to an ordered list of Blocks.

    Mapping rules
    -------------
    * ``system``  (str or list-of-content blocks) → one STABLE ``Kind.SYSTEM``
      block with id ``"system"``.
    * ``tools``   (list of tool dicts)            → one STABLE ``Kind.TOOLS``
      block with id ``"tools"`` (json-serialised).
    * ``messages``:
      - user / text content → ``Kind.USER`` VOLATILE  id ``msg{i}``
      - assistant / text content → ``Kind.HISTORY`` SETTLING  id ``msg{i}``
      - ``tool_result`` content block → ``Kind.TOOL_OUTPUT`` VOLATILE  id ``msg{i}``
      - images and ``tool_use`` inputs are ignored.
    """
    blocks: list[Block] = []

    # --- system ---
    system_raw = body.get("system")
    if system_raw is not None:
        if isinstance(system_raw, str):
            system_text = system_raw
        else:
            # list of content blocks — join all text parts
            parts = [
                c["text"]
                for c in system_raw
                if isinstance(c, dict)
                and c.get("type") == "text"
                and isinstance(c.get("text"), str)
            ]
            system_text = "\n".join(parts)
        blocks.append(
            Block(
                id="system",
                kind=Kind.SYSTEM,
                text=system_text,
                stability=Stability.STABLE,
                decision_relevant=False,
            )
        )

    # --- tools ---
    tools_raw = body.get("tools")
    if tools_raw:
        blocks.append(
            Block(
                id="tools",
                kind=Kind.TOOLS,
                text=json.dumps(tools_raw, separators=(",", ":")),
                stability=Stability.STABLE,
                decision_relevant=False,
            )
        )

    # --- messages ---
    messages = body.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        block_id = f"msg{i}"

        if isinstance(content, str):
            # simple string body
            if role == "user":
                blocks.append(
                    Block(
                        id=block_id,
                        kind=Kind.USER,
                        text=content,
                        stability=Stability.VOLATILE,
                        decision_relevant=False,
                    )
                )
            elif role == "assistant":
                blocks.append(
                    Block(
                        id=block_id,
                        kind=Kind.HISTORY,
                        text=content,
                        stability=Stability.SETTLING,
                        decision_relevant=False,
                    )
                )
        else:
            # list of content blocks
            content_blocks = content if isinstance(content, list) else []
            sub = 0
            for cb in content_blocks:
                if not isinstance(cb, dict):
                    continue
                ctype = cb.get("type")
                if ctype == "text":
                    text = cb.get("text", "")
                    if role == "user":
                        blocks.append(
                            Block(
                                id=f"{block_id}_{sub}",
                                kind=Kind.USER,
                                text=text,
                                stability=Stability.VOLATILE,
                                decision_relevant=False,
                            )
                        )
                    elif role == "assistant":
                        blocks.append(
                            Block(
                                id=f"{block_id}_{sub}",
                                kind=Kind.HISTORY,
                                text=text,
                                stability=Stability.SETTLING,
                                decision_relevant=False,
                            )
                        )
                    sub += 1
                elif ctype == "tool_result":
                    # Extract text from tool_result content
                    result_content = cb.get("content", "")
                    if isinstance(result_content, str):
                        result_text = result_content
                    elif isinstance(result_content, list):
                        parts = [
                            rc["text"]
                            for rc in result_content
                            if isinstance(rc, dict)
                            and rc.get("type") == "text"
                            and isinstance(rc.get("text"), str)
                        ]
                        result_text = "\n".join(parts)
                    else:
                        result_text = str(result_content)
                    blocks.append(
                        Block(
                            id=f"{block_id}_{sub}",
                            kind=Kind.TOOL_OUTPUT,
                            text=result_text,
                            stability=Stability.VOLATILE,
                            decision_relevant=False,
                        )
                    )
                    sub += 1
                # images and tool_use are intentionally skipped

    return blocks


# ---------------------------------------------------------------------------
# OpenAI Chat Completions API
# ---------------------------------------------------------------------------


def ingest_openai_request(body: dict, *, turn_index: int = 0) -> list[Block]:
    """Map one OpenAI Chat Completions request body to an ordered list of Blocks.

    Mapping rules
    -------------
    * ``tools`` (list) → one STABLE ``Kind.TOOLS`` block with id ``"tools"``.
    * ``messages``:
      - role ``system``    → ``Kind.SYSTEM`` STABLE      id ``"system"``
      - role ``user``      → ``Kind.USER`` VOLATILE       id ``msg{i}``
      - role ``assistant`` → ``Kind.HISTORY`` SETTLING    id ``msg{i}``
      - role ``tool``      → ``Kind.TOOL_OUTPUT`` VOLATILE id ``msg{i}``
      - images and ``tool_calls`` in content are ignored.
    """
    blocks: list[Block] = []

    # --- tools (added before messages, before system to keep stable prefix first) ---
    tools_raw = body.get("tools")
    system_block: Block | None = None

    # We'll collect system + tools in stable order, then messages
    messages = body.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        block_id = f"msg{i}"

        if role == "system":
            text = _openai_content_to_text(content)
            system_block = Block(
                id="system",
                kind=Kind.SYSTEM,
                text=text,
                stability=Stability.STABLE,
                decision_relevant=False,
            )
            # Don't append yet — prepend at the start
        elif role == "user":
            text = _openai_content_to_text(content)
            if text:
                blocks.append(
                    Block(
                        id=block_id,
                        kind=Kind.USER,
                        text=text,
                        stability=Stability.VOLATILE,
                        decision_relevant=False,
                    )
                )
        elif role == "assistant":
            text = _openai_content_to_text(content)
            if text:
                blocks.append(
                    Block(
                        id=block_id,
                        kind=Kind.HISTORY,
                        text=text,
                        stability=Stability.SETTLING,
                        decision_relevant=False,
                    )
                )
        elif role == "tool":
            text = _openai_content_to_text(content)
            blocks.append(
                Block(
                    id=block_id,
                    kind=Kind.TOOL_OUTPUT,
                    text=text,
                    stability=Stability.VOLATILE,
                    decision_relevant=False,
                )
            )

    # Build the final ordered list: system → tools → message blocks
    prefix: list[Block] = []
    if system_block is not None:
        prefix.append(system_block)
    if tools_raw:
        prefix.append(
            Block(
                id="tools",
                kind=Kind.TOOLS,
                text=json.dumps(tools_raw, separators=(",", ":")),
                stability=Stability.STABLE,
                decision_relevant=False,
            )
        )

    return prefix + blocks


def _openai_content_to_text(content: object) -> str:
    """Flatten OpenAI message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
            # image_url and other types are skipped
        return "\n".join(parts)
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Session / file ingestion
# ---------------------------------------------------------------------------


def ingest_session(
    requests: list[dict],
    *,
    provider: str = "anthropic",
    id: str = "ingested",
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """Convert a list of provider request bodies into a multi-turn Trajectory.

    Each request becomes one Turn.  Successive requests in a real session share
    a growing prefix (system + tools), which is exactly the pattern that
    cache-aware simulation rewards.

    Parameters
    ----------
    requests:
        Ordered list of raw request body dicts (as captured from the wire).
    provider:
        ``"anthropic"`` or ``"openai"``.
    id:
        Trajectory id string.
    model:
        Model identifier used for pricing lookups.
    """
    if provider == "anthropic":
        ingest_fn = ingest_anthropic_request
    elif provider == "openai":
        ingest_fn = ingest_openai_request
    else:
        raise ValueError(f"unknown provider {provider!r}; expected 'anthropic' or 'openai'")

    turns: list[Turn] = []
    for i, req in enumerate(requests):
        blocks = ingest_fn(req, turn_index=i)
        turns.append(Turn(index=i, blocks=blocks))

    return Trajectory(id=id, model=model, turns=turns)


def ingest_file(
    path: str,
    *,
    provider: str = "anthropic",
    model: str = DEFAULT_MODEL,
) -> Trajectory:
    """Load a file of captured request bodies and build a Trajectory.

    Supported formats
    -----------------
    * ``.json``  — either a single request body dict or a list of dicts.
    * ``.jsonl`` — one JSON object per line (blank lines are skipped).

    The trajectory id is derived from the file stem.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"distil ingest: input file not found: {path}")
    suffix = p.suffix.lower()
    traj_id = p.stem

    if suffix == ".jsonl":
        requests: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                requests.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue  # skip a malformed line rather than abort the whole ingest
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            requests = raw
        else:
            requests = [raw]

    return ingest_session(requests, provider=provider, id=traj_id, model=model)
