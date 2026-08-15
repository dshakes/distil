# The paper (`main.tex`)

An arXiv-ready LaTeX source for *Certified Decision-Equivalent Context Compression
for LLM Agents*. **Every figure is TikZ/pgfplots** — no external image files, so it
compiles anywhere with a standard TeX distribution.

## Compile

**Easiest (no install) — Overleaf:**
1. Go to [overleaf.com](https://www.overleaf.com) → *New Project* → *Upload Project*,
   and upload `main.tex` (or this whole `docs/paper/` folder).
2. Set the compiler to **pdfLaTeX** (Menu → Compiler). Click *Recompile*.

**Local:**
```bash
# needs a TeX distribution (TeX Live / MacTeX), with tikz + pgfplots + algorithm2e
latexmk -pdf main.tex      # or: pdflatex main.tex (run twice for cross-refs)
```

**Committing a rebuilt PDF:** `main` runs a byte-for-byte staleness check against
`main.pdf`/`main_neurips.pdf`, so a source edit needs the compiled PDF committed
alongside it. Match CI's build pin (see `paper-build.yml`) exactly or the check
will flag your own unchanged content as stale — `-g` forces a full recompile
(otherwise `latexmk` may see the committed PDF as already up-to-date and skip
rebuilding), `SOURCE_DATE_EPOCH` pins the PDF's own `/CreationDate`/`/ID`, and
`FORCE_SOURCE_DATE=1` is required *in addition* — without it `\date{\today}` in
both papers still renders today's real date, since `\today` reads TeX's
`\year`/`\month`/`\day` primitives rather than the PDF metadata:
```bash
# macOS/Linux
cd docs/paper && SOURCE_DATE_EPOCH=1700000000 FORCE_SOURCE_DATE=1 latexmk -pdf -g main.tex main_neurips.tex \
  && git add main.pdf main_neurips.pdf

# Windows (PowerShell)
cd docs/paper; $env:SOURCE_DATE_EPOCH=1700000000; $env:FORCE_SOURCE_DATE=1; latexmk -pdf -g main.tex main_neurips.tex
git add main.pdf main_neurips.pdf
```

## Filling the headline numbers

The result macros at the top of `main.tex` (`\HLsavings`, `\HLcoverage`, `\HLrisk`)
and the `pgfplots` coordinates in §Results are placeholders/illustrative. Replace
them with values from a real run:

```bash
python benchmarks/prove.py --dataset tau --path tau.json \
   --runner anthropic --model claude-opus-4-8 --samples 3 --expand \
   --alpha 0.05 --delta 0.05 --ladder full --reps 500 --report results.json
```
Then copy the E1 frontier points, E2 coverage, and E4 table out of `results.json`
into the corresponding figures/tables.

## Switching to a venue style

Replace `\documentclass[11pt]{article}` with the venue's style file (e.g.
`neurips_2025.sty`, `icml2025.sty`, `acl.sty`) and keep the body. Most ML venues use
a two-column or single-column style with their own title block; move `\author` into
their macro.
