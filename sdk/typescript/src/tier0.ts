/**
 * Tier-0 lossless transforms — strict TypeScript port of distil/compress/tier0.py.
 *
 * BYTE-IDENTITY CONTRACT
 * ----------------------
 * Every function in this file targets output that is byte-identical to the
 * equivalent Python function, and that contract is enforced by
 * tests/test_ts_conformance.py (Python) and the golden-parity suite here.
 * Where byte-identity is not achievable (see `minifyJson`), the transform
 * DECLINES rather than producing uncertified output.
 *
 * Tier-1 (reversible digest) is deliberately NOT implemented here: it allocates
 * restore handles that must share one store with the proxy and the MCP server.
 * Route through the distil proxy for digest-tier savings.
 */

// Matches tier0.py's _RUN_MARKER: a generated marker, so collapsing a run of
// markers would make the new marker indistinguishable from original content.
const RUN_MARKER = /^<<x\d+>>$/;

/**
 * Run-length-encode consecutive identical lines, reversibly.
 *
 * Two or more identical newline-terminated lines `L` collapse to `L\n<<xN>>\n`
 * where N is the exact original count — so the input is fully recoverable.
 * Lines that already look like run markers are left alone.
 *
 * Mirrors Python `distil.compress.tier0.collapse_runs`.
 */
export function collapseRuns(text: string): string {
  if (!text.includes("\n")) return text;

  const lines = text.split("\n");
  const out: string[] = [];
  let i = 0;
  // The Python regex matches only newline-TERMINATED runs. Splitting produces a
  // trailing empty element for a text ending in \n; the last element of a text
  // NOT ending in \n has no terminator and cannot be in a run.
  const lastIndex = lines.length - 1;
  while (i < lines.length) {
    const line = lines[i] as string;
    let j = i;
    // Count consecutive newline-terminated copies.
    while (j < lastIndex && lines[j] === line) j++;
    const runLen = j - i;
    if (runLen >= 2 && !RUN_MARKER.test(line)) {
      out.push(line, `<<x${runLen}>>`);
      i = j;
    } else {
      out.push(line);
      i += 1;
    }
  }
  return out.join("\n");
}

/**
 * Textual inverse of `collapseRuns`.
 *
 * PRECONDITION: the original text contained no line matching /^<<x\d+>>$/.
 * A generated marker and original content that looks like one are
 * indistinguishable — `collapseRuns` refuses to collapse such runs precisely
 * so they are never lost. Use the restore store for recovery, not this function.
 */
export function expandRuns(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const m = /^<<x(\d+)>>$/.exec(lines[i] as string);
    if (m !== null && out.length > 0) {
      const prev = out[out.length - 1] as string;
      const n = parseInt(m[1] as string, 10);
      for (let k = 1; k < n; k++) out.push(prev);
    } else {
      out.push(lines[i] as string);
    }
  }
  return out.join("\n");
}

// ── JSON minification ────────────────────────────────────────────────────────
//
// Python's json.dumps and JS's JSON.stringify disagree on:
//   1. INTEGRAL FLOATS: Python renders 1.0 as "1.0"; JS renders it "1".
//   2. KEY ORDER: JS objects hoist integer-like keys above string keys.
//
// Both are detected below and cause the transform to DECLINE (return null),
// keeping the caller's text byte-exact — the same posture as tier0.py for
// duplicate keys and non-finite numbers.

/** True when a key would be reordered by JS object internals. */
function isIntegerLikeKey(k: string): boolean {
  return /^(0|[1-9]\d*)$/.test(k) && Number(k) <= 4_294_967_294;
}

/** Replace every JSON string literal with "" so numeric scans see structure only. */
function stripJsonStrings(src: string): string {
  let out = "";
  let inStr = false;
  let esc = false;
  for (let i = 0; i < src.length; i++) {
    const c = src[i] as string;
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') {
        inStr = false;
        out += '""';
      }
      continue;
    }
    if (c === '"') inStr = true;
    else out += c;
  }
  return out;
}

/**
 * Scan the raw JSON source for constructs we cannot round-trip byte-identically
 * with Python. Returns a reason string, or null when it is safe to proceed.
 */
function unsafeJsonReason(src: string): string | null {
  const bare = stripJsonStrings(src);
  // The fraction is optional so every match consumes a whole digit run: linear
  // time, no retry from each start position of a long digit string (ReDoS).
  const nums = bare.match(/-?\d+(?:\.\d+)?/g) ?? [];
  for (const n of nums) {
    if (n.includes(".") && Number.isInteger(Number(n))) {
      return "integral float renders differently in Python and JS";
    }
  }
  if (/\d[eE][+-]?\d+/.test(bare)) return "exponent notation renders differently";
  if (/\b(Infinity|-Infinity|NaN)\b/.test(bare)) return "non-finite constant";
  return null;
}

/** Depth-first check for integer-like keys (which JS objects reorder). */
function objectKeysAreSafe(obj: unknown): boolean {
  if (Array.isArray(obj)) return obj.every((v) => objectKeysAreSafe(v));
  if (obj !== null && typeof obj === "object") {
    for (const k of Object.keys(obj)) {
      if (isIntegerLikeKey(k)) return false;
      if (!objectKeysAreSafe((obj as Record<string, unknown>)[k])) return false;
    }
  }
  return true;
}

/** Cheap duplicate-key detector over raw JSON source. */
function hasDuplicateKeys(src: string): boolean {
  const stack: Set<string>[] = [];
  let inStr = false;
  let esc = false;
  let cur = 0;
  let pendingKey: string | null = null;
  for (let i = 0; i < src.length; i++) {
    const c = src[i] as string;
    if (inStr) {
      if (esc) {
        esc = false;
      } else if (c === "\\") {
        esc = true;
      } else if (c === '"') {
        inStr = false;
        pendingKey = src.slice(cur + 1, i);
      }
      continue;
    }
    if (c === '"') {
      inStr = true;
      cur = i;
    } else if (c === "{") {
      stack.push(new Set<string>());
      pendingKey = null;
    } else if (c === "}") {
      stack.pop();
      pendingKey = null;
    } else if (c === ":" && stack.length > 0 && pendingKey !== null) {
      const top = stack[stack.length - 1] as Set<string>;
      if (top.has(pendingKey)) return true;
      top.add(pendingKey);
      pendingKey = null;
    } else if (c === ",") {
      pendingKey = null;
    }
  }
  return false;
}

/**
 * Re-encode JSON with no incidental whitespace, or null when the round-trip
 * cannot be proven byte-identical to the Python implementation.
 *
 * Mirrors `distil.compress.tier0.minify_json`. When this returns null the
 * caller keeps the text byte-exact.
 */
export function minifyJson(text: string): string | null {
  const s = text.trim();
  const first = s.charAt(0);
  const last = s.charAt(s.length - 1);
  if (!((first === "{" || first === "[") && (last === "}" || last === "]"))) {
    return null;
  }
  if (unsafeJsonReason(s) !== null) return null;

  let obj: unknown;
  try {
    obj = JSON.parse(s);
  } catch {
    return null;
  }
  if (!objectKeysAreSafe(obj)) return null;
  if (hasDuplicateKeys(s)) return null;

  try {
    return JSON.stringify(obj);
  } catch {
    return null; // circular refs cannot come from JSON.parse, but never throw
  }
}

// ── Heuristic tokenizer ──────────────────────────────────────────────────────
// Port of distil/tokenizer.py:HeuristicTokenizer.
// Pattern: \w+|[^\w\s] in Unicode mode — one word-char run OR one non-word
// non-space char. The `u` flag + Unicode property classes reproduce Python's
// \w/\s semantics for counting purposes.

const PIECE = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;
const SUBWORD_FACTOR = 1.33;

/**
 * Round-half-to-even (banker's rounding), matching Python's built-in `round()`.
 * JS `Math.round` is round-half-up and silently diverges at exact 0.5 boundaries.
 */
function bankersRound(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Approximate token count, matching Python `HeuristicTokenizer.count`.
 * Returns 0 for empty/falsy input; at least 1 for any non-empty non-whitespace.
 */
export function countTokens(text: string): number {
  if (!text) return 0;
  const pieces = text.match(PIECE);
  const n = pieces !== null ? pieces.length : 0;
  return Math.max(1, bankersRound(n * SUBWORD_FACTOR));
}

/**
 * Apply every Tier-0 transform in production order (mirrors
 * `distil.adapters.anthropic._apply_tier0`):
 *   1. JSON minification (byte-safe subset only)
 *   2. Run-collapse, kept only when it does not increase the token count
 *
 * Tier-0 must NEVER inflate: if the collapsed form costs more tokens than the
 * original, the original is returned.
 */
export function applyTier0(text: string): string {
  if (!text) return text;
  const mj = minifyJson(text);
  const base = mj !== null ? mj : text;
  const collapsed = collapseRuns(base);
  if (collapsed !== base && countTokens(collapsed) <= countTokens(base)) {
    return collapsed;
  }
  return base;
}
