// Tier-0 lossless transforms — a faithful port of distil/compress/tier0.py.
//
// WHY THIS IS A PORT AND NOT A REIMPLEMENTATION
// ---------------------------------------------
// distil's whole claim is a decision-equivalence certificate, and that
// certificate is measured against the PYTHON compressor. A JS implementation
// that merely "compresses similarly" would sit outside it — users would get a
// guarantee that does not describe the code they are running.
//
// So this file targets BYTE-IDENTICAL output with tier0.py, and
// tests/test_ts_conformance.py runs both against a shared fixture corpus and
// fails on any divergence. Where byte-identity is not achievable (see
// `minifyJson`), the transform DECLINES rather than producing output the
// certificate does not cover. Declining costs a little compression; diverging
// silently would cost the guarantee.
//
// Tier-1 (the reversible digest) is deliberately NOT ported: it allocates
// restore handles and must share one store with the proxy and the MCP server.
// Use `compressViaProxy` in compress.js for that.

"use strict";

// Matches tier0.py's _RUN_MARKER: a generated marker, so a run of markers is
// left alone (collapsing it would make the new marker indistinguishable from
// original content when reversing).
const RUN_MARKER = /^<<x\d+>>$/;

/**
 * Run-length-encode consecutive identical lines, reversibly.
 * Two or more identical lines `L` collapse to `L\n<<xN>>\n`, where N is the
 * exact original count, so the input is fully recoverable.
 *
 * Pure string manipulation, so this is exactly portable — no locale, no number
 * formatting, no object ordering. It is the bulk of Tier-0's real-world win
 * (repeated log lines, duplicated stack frames).
 *
 * @param {string} text
 * @returns {string}
 */
function collapseRuns(text) {
  if (typeof text !== "string" || text.indexOf("\n") === -1) return text;

  const lines = text.split("\n");
  const out = [];
  let i = 0;
  // The Python side uses the regex /^(.*)\n(?:\1\n)+/m, which only ever matches
  // runs that are newline-TERMINATED. A trailing final line with no newline
  // after it therefore cannot participate in a run; splitting keeps that
  // property because the last element has no terminator to consume.
  const lastIndex = lines.length - 1;
  while (i < lines.length) {
    const line = lines[i];
    let j = i;
    // Count how many consecutive newline-terminated copies follow.
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
 * PRECONDITION: the ORIGINAL text contained no line matching /^<<x\d+>>$/.
 * A generated marker and original content that looks like one are
 * indistinguishable after the fact, so no textual inverse can separate them.
 * `collapseRuns` refuses to collapse such runs precisely so nothing is lost —
 * but expanding text that already contained markers will invent lines.
 *
 * This is why distil's real reversibility guarantee comes from the restore
 * store (the exact original bytes, keyed by handle) rather than from inverting
 * the transform. Use this for local verification of ordinary text, not as the
 * recovery path.
 *
 * @param {string} text
 * @returns {string}
 */
function expandRuns(text) {
  if (typeof text !== "string") return text;
  const lines = text.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const m = /^<<x(\d+)>>$/.exec(lines[i]);
    if (m && out.length > 0) {
      const prev = out[out.length - 1];
      const n = parseInt(m[1], 10);
      for (let k = 1; k < n; k++) out.push(prev);
    } else {
      out.push(lines[i]);
    }
  }
  return out.join("\n");
}

// --- JSON ------------------------------------------------------------------
//
// Python's json.dumps and JS's JSON.stringify disagree on two things that
// matter for byte-identity:
//
//   1. NUMBER FORMATTING. Python renders a float that happens to be integral as
//      "1.0"; JS renders it "1". Nothing in JS can recover the distinction,
//      because JSON.parse has already collapsed 1.0 and 1 to the same double.
//   2. KEY ORDER. Python dicts preserve insertion order for every key. JS
//      objects hoist integer-like keys ("2" before "a", regardless of input).
//
// Both are detected below and cause the transform to DECLINE (return null), so
// the caller keeps the text byte-exact. That is the same posture tier0.py takes
// for duplicate keys and non-finite numbers: when the round trip is not
// provably value-preserving, do nothing.

/** True when a key would be reordered by a JS object (array-index-like). */
function isIntegerLikeKey(k) {
  return /^(0|[1-9]\d*)$/.test(k) && Number(k) <= 4294967294;
}

/**
 * Scan the raw JSON source for constructs we cannot round-trip byte-identically
 * with Python. Returns a reason string, or null when it is safe.
 */
function unsafeJsonReason(src) {
  // Strip string contents FIRST. Scanning raw source treats `"line1"` as
  // exponent notation ("e1") and `"v1.0"` as an integral float, which made this
  // decline JSON that Python happily minifies — a conformance divergence in the
  // safe direction, but a divergence.
  const bare = stripJsonStrings(src);

  // Python renders an integral float as "1.0"; JS renders it "1". After parsing
  // the distinction is gone, so it has to be decided on the literal.
  const nums = bare.match(/-?\d+\.\d+/g) || [];
  for (const n of nums) {
    if (Number.isInteger(Number(n))) {
      return "integral float renders differently in Python and JS";
    }
  }
  if (/\d[eE][+-]?\d+/.test(bare)) return "exponent notation renders differently";
  if (/\b(Infinity|-Infinity|NaN)\b/.test(bare)) return "non-finite constant";
  return null;
}

/** Replace every JSON string literal with "" so scans see structure only. */
function stripJsonStrings(src) {
  let out = "";
  let inStr = false;
  let esc = false;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
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

/** Depth-first check for duplicate or reorder-prone keys, on the raw source. */
function objectKeysAreSafe(obj, seenPath) {
  if (Array.isArray(obj)) return obj.every((v) => objectKeysAreSafe(v, seenPath));
  if (obj && typeof obj === "object") {
    for (const k of Object.keys(obj)) {
      if (isIntegerLikeKey(k)) return false; // JS would hoist it
      if (!objectKeysAreSafe(obj[k], seenPath)) return false;
    }
  }
  return true;
}

/**
 * Re-encode JSON with no incidental whitespace, or null when that would not be
 * provably value-preserving *and* byte-identical to the Python implementation.
 *
 * @param {string} text
 * @returns {string|null}
 */
function minifyJson(text) {
  if (typeof text !== "string") return null;
  const s = text.trim();
  const first = s.charAt(0);
  const last = s.charAt(s.length - 1);
  // Same gate as tier0.py: only whole objects/arrays.
  if (!((first === "{" || first === "[") && (last === "}" || last === "]"))) return null;

  if (unsafeJsonReason(s) !== null) return null;

  let obj;
  try {
    obj = JSON.parse(s);
  } catch (_e) {
    return null;
  }
  if (!objectKeysAreSafe(obj)) return null;

  // Duplicate keys: JSON.parse silently keeps the last. tier0.py rejects such
  // input outright, so detect it by counting keys in the source.
  if (hasDuplicateKeys(s)) return null;

  try {
    return JSON.stringify(obj);
  } catch (_e) {
    return null; // circular refs cannot occur from JSON.parse, but never throw
  }
}

/** Cheap duplicate-key detector over the raw source (string-aware). */
function hasDuplicateKeys(src) {
  const stack = [];
  let inStr = false;
  let esc = false;
  let cur = null;
  let pendingKey = null;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
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
      stack.push(new Set());
      pendingKey = null;
    } else if (c === "}") {
      stack.pop();
      pendingKey = null;
    } else if (c === ":" && stack.length && pendingKey !== null) {
      const top = stack[stack.length - 1];
      if (top.has(pendingKey)) return true;
      top.add(pendingKey);
      pendingKey = null;
    } else if (c === ",") {
      pendingKey = null;
    }
  }
  return false;
}

// --- heuristic tokenizer ----------------------------------------------------
// Port of distil/tokenizer.py:HeuristicTokenizer. Needed because the production
// Tier-0 path is reject-if-bigger *by tokens*, not by characters: collapsing a
// run of near-free blank lines into a "<<xN>>" marker can cost MORE tokens than
// it removes, and Tier-0 must never inflate.

// Python: re.compile(r"\w+|[^\w\s]", re.UNICODE). The `u` flag plus explicit
// Unicode property escapes reproduce Python's \w/\s semantics closely enough
// for counting; the conformance suite is what proves it on real inputs.
const PIECE = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;
const SUBWORD_FACTOR = 1.33;

/**
 * Approximate token count, matching HeuristicTokenizer.count byte-for-byte.
 * @param {string} text
 * @returns {number}
 */
function countTokens(text) {
  if (!text) return 0;
  const pieces = text.match(PIECE);
  const n = pieces ? pieces.length : 0;
  return Math.max(1, bankersRound(n * SUBWORD_FACTOR));
}

/**
 * Python's round() is round-half-to-EVEN; JS Math.round is round-half-UP.
 * They differ on exact .5, which n * 1.33 can land on, so the naive version
 * would silently diverge from Python on specific input lengths.
 */
function bankersRound(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Apply every Tier-0 transform in the same order as the production path
 * (distil/adapters/anthropic.py:_apply_tier0): JSON minification first, then
 * run-collapse on the RESULT, keeping the collapse only when it does not
 * increase the token count.
 *
 * @param {string} text
 * @returns {string}
 */
function applyTier0(text) {
  if (typeof text !== "string" || text.length === 0) return text;
  const mj = minifyJson(text);
  const base = mj !== null ? mj : text;
  const collapsed = collapseRuns(base);
  if (collapsed !== base && countTokens(collapsed) <= countTokens(base)) {
    return collapsed;
  }
  return base;
}

module.exports = {
  applyTier0,
  countTokens,
  collapseRuns,
  expandRuns,
  minifyJson,
  // exported for the conformance harness
  _internals: { hasDuplicateKeys, unsafeJsonReason, isIntegerLikeKey },
};
