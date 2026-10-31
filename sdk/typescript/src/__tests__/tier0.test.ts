/**
 * Unit tests for Tier-0 transforms.
 *
 * Covers the same cases as tests/test_ts_conformance.py (Python ↔ JS parity)
 * plus the TS-specific type guards.
 */
import { describe, expect, it } from "vitest";
import { applyTier0, collapseRuns, countTokens, expandRuns, minifyJson } from "../tier0.js";

// ── collapseRuns ──────────────────────────────────────────────────────────────

describe("collapseRuns", () => {
  it("leaves non-repeating text unchanged", () => {
    expect(collapseRuns("a\nb\nc\n")).toBe("a\nb\nc\n");
  });
  it("collapses 2 identical lines", () => {
    expect(collapseRuns("a\na\n")).toBe("a\n<<x2>>\n");
  });
  it("collapses 3 identical lines", () => {
    expect(collapseRuns("a\na\na\n")).toBe("a\n<<x3>>\n");
  });
  it("collapses 4 identical lines", () => {
    expect(collapseRuns("a\na\na\na\n")).toBe("a\n<<x4>>\n");
  });
  it("handles mixed runs", () => {
    expect(collapseRuns("a\na\na\nb\nb\nb\nc")).toBe("a\n<<x3>>\nb\n<<x3>>\nc");
  });
  it("does not collapse when last line has no trailing newline", () => {
    expect(collapseRuns("a\na")).toBe("a\na");
  });
  it("handles empty string", () => {
    expect(collapseRuns("")).toBe("");
  });
  it("handles single line", () => {
    expect(collapseRuns("hello\n")).toBe("hello\n");
  });
  it("collapses empty lines", () => {
    expect(collapseRuns("\n\n\n")).toBe("\n<<x3>>\n");
  });
  it("handles interleaved runs", () => {
    expect(collapseRuns("x\nx\ny\nz\nz\nz\nz\n")).toBe("x\n<<x2>>\ny\nz\n<<x4>>\n");
  });
  it("refuses to collapse runs of run markers", () => {
    const text = "<<x3>>\n<<x3>>\n<<x3>>\nreal\n";
    expect(collapseRuns(text)).toBe(text);
  });
  it("matches the large-log corpus case", () => {
    const text = "ERROR failed to connect\n".repeat(50) + "done\n";
    const collapsed = collapseRuns(text);
    expect(collapsed).toContain("<<x50>>");
    expect(collapsed.length).toBeLessThan(text.length);
  });
});

// ── expandRuns ────────────────────────────────────────────────────────────────

describe("expandRuns", () => {
  it("round-trips plain text", () => {
    const text = "a\na\na\nb\nc\nc\n";
    expect(expandRuns(collapseRuns(text))).toBe(text);
  });
  it("round-trips unicode", () => {
    const text = "héllo wörld ✓\nhéllo wörld ✓\n";
    expect(expandRuns(collapseRuns(text))).toBe(text);
  });
  it("round-trips mixed runs", () => {
    const text = "x\nx\ny\nz\nz\nz\nz\n";
    expect(expandRuns(collapseRuns(text))).toBe(text);
  });
  it("round-trips large log", () => {
    const text = "ERROR failed to connect\n".repeat(50) + "done\n";
    expect(expandRuns(collapseRuns(text))).toBe(text);
  });
});

// ── minifyJson ────────────────────────────────────────────────────────────────

describe("minifyJson", () => {
  it("minifies a simple object", () => {
    expect(minifyJson('{"b": 1, "a": 2}')).toBe('{"b":1,"a":2}');
  });
  it("minifies a simple array", () => {
    expect(minifyJson("[1, 2, 3]")).toBe("[1,2,3]");
  });
  it("minifies a nested structure", () => {
    expect(minifyJson('{"outer": {"inner": [1, 2, 3], "s": "x"}}')).toBe(
      '{"outer":{"inner":[1,2,3],"s":"x"}}',
    );
  });
  it("preserves unicode", () => {
    expect(minifyJson('{"k": "héllo ✓"}')).toBe('{"k":"héllo ✓"}');
  });
  it("returns null for non-JSON", () => {
    expect(minifyJson("{this is not json}")).toBeNull();
  });
  it("returns null for partial JSON", () => {
    expect(minifyJson('{"a": 1')).toBeNull();
  });
  it("returns null for integral float", () => {
    expect(minifyJson('{"a": 1.0}')).toBeNull();
  });
  it("returns null for nested integral float", () => {
    expect(minifyJson('{"a": [2.0, 3.5]}')).toBeNull();
  });
  it("returns null for exponent notation", () => {
    expect(minifyJson('{"a": 1e5}')).toBeNull();
  });
  it("returns null for integer-like keys (JS reorders them)", () => {
    expect(minifyJson('{"2": "b", "1": "a"}')).toBeNull();
  });
  it("returns null for duplicate keys", () => {
    expect(minifyJson('{"a": 1, "a": 2}')).toBeNull();
  });
  it("returns null for duplicate keys spelled with escapes", () => {
    expect(minifyJson('{"a": 1, "\\u0061": 2}')).toBeNull();
  });
  it("does not mistake text inside strings for numbers (line1 → not exponent)", () => {
    expect(minifyJson('{"msg": "line1"}')).toBe('{"msg":"line1"}');
  });
  it("handles escaped quotes inside strings", () => {
    expect(minifyJson('{"msg": "line1\\nline2", "esc": "quote\\" here"}')).toBe(
      '{"msg":"line1\\nline2","esc":"quote\\" here"}',
    );
  });
});

// ── countTokens ───────────────────────────────────────────────────────────────

describe("countTokens", () => {
  it("returns 0 for empty string", () => {
    expect(countTokens("")).toBe(0);
  });
  it("returns at least 1 for any non-empty text", () => {
    expect(countTokens("a")).toBeGreaterThanOrEqual(1);
  });
  it("matches known hello-world value", () => {
    // 2 word pieces → round(2 × 1.33) = round(2.66) = 3
    expect(countTokens("hello world")).toBe(3);
  });
  it("counts punctuation as separate pieces", () => {
    // hello , world ! = 4 pieces → round(4 × 1.33) = round(5.32) = 5
    expect(countTokens("hello, world!")).toBe(5);
  });
  it("handles unicode letters as word chars", () => {
    // "héllo" is 1 piece → round(1 × 1.33) = 1
    expect(countTokens("héllo")).toBe(1);
  });
  it("matches corpus parity: foo bar baz qux = 5", () => {
    // 4 pieces × 1.33 = 5.32 → 5
    expect(countTokens("foo bar baz qux")).toBe(5);
  });
  it("handles whitespace-only: 0 pieces → max(1, 0) = 1", () => {
    expect(countTokens("   ")).toBe(1);
  });
});

// ── applyTier0 ────────────────────────────────────────────────────────────────

describe("applyTier0", () => {
  it("compresses a repetitive log", () => {
    const text = "ERROR failed to connect\n".repeat(50) + "done\n";
    const result = applyTier0(text);
    expect(result.length).toBeLessThan(text.length);
  });
  it("never inflates: a tiny run is not worth a marker", () => {
    // "a\na\na\n" → collapsed "a\n<<x3>>\n" costs more tokens than a tiny run
    const tiny = "a\na\na\n";
    expect(applyTier0(tiny)).toBe(tiny);
  });
  it("minifies JSON", () => {
    expect(applyTier0('{"b": 1, "a": 2}')).toBe('{"b":1,"a":2}');
  });
  it("handles empty string", () => {
    expect(applyTier0("")).toBe("");
  });
  it("leaves non-JSON text unchanged when no runs", () => {
    const t = "just one line\n";
    expect(applyTier0(t)).toBe(t);
  });
});
