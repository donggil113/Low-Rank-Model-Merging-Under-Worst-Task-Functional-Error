# Building the working draft

## Style files (official, unmodified)

- Source: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip, linked from
  https://iclr.cc/Conferences/2027/AuthorGuidelines. Fetched 2026-09-26.
- ZIP sha256: `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
- Files copied unmodified into `paper/`: `iclr2027_conference.sty`, `iclr2027_conference.bst`,
  `fancyhdr.sty`, `natbib.sty`, `math_commands.tex`. Per-file hashes are in
  `STYLE_SHA256.txt`; each copy is identical to the file inside the ZIP.
- Anonymous mode is used (`\iclrfinalcopy` is not set). The style prints
  "Under review as a conference paper at ICLR 2027" in the running header. This draft has not
  been submitted, so `main.tex` replaces the header with `\lhead{Internal working draft v0 --- not
  submitted ...}` right after `\maketitle`. The `.sty` file itself is not edited. The anonymous
  author block ("Anonymous authors / Paper under double-blind review") is printed by the style and
  is kept. A boxed status note on page 1 states that the paper is not submitted.

## Generated inputs (run before building)

```bash
python3 paper/tools/make_bib.py        # references.bib from paper/bib_sources/ (fetched metadata)
python3 paper/tools/export_results.py  # generated/numbers.tex, generated/provenance.tsv, tables/, figures/
```

## Compile

```bash
python3 paper/tools/build.py           # see below; records CPU time to paper/build_log.jsonl
```

The build status (engine, version, pages, warnings) is recorded in `PAPER_STATUS.md` after each
build. The build-and-check CPU budget is 600 s. It is tracked separately from the analysis budget.
