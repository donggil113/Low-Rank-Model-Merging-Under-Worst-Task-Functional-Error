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

## Toolchain (installed for this draft)

No LaTeX engine was present. Tectonic could not be obtained: GitHub release downloads returned
403 through the session proxy. The minimal TeX Live subset was therefore installed from the signed
Ubuntu 24.04 archive with `--no-install-recommends`.

- Packages: `texlive-latex-base`, `texlive-latex-recommended`, `texlive-fonts-recommended`,
  `texlive-pictures`, `poppler-utils` (for `pdfinfo`/`pdftoppm` page checks). Exact versions are in
  `build_env/installed_packages.tsv`.
- Scope: 20 packages were newly installed. One existing package, `libpoppler134`, was upgraded as
  a dependency. The install used about 234 MB of disk.
- Attempt 1 FAILED with exit 100 because the local package index was stale (404 on poppler-utils).
  The log is preserved in `build_env/apt_install_attempt1_FAILED.log`.
- Attempt 2 ran `apt-get update` and then the install, and succeeded. It took 52.3 s wall time and
  52.2 s CPU (`build_env/apt_install_attempt2_cost.json`).
- Engine: pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian); BibTeX from the same distribution.

## Compile

```bash
python3 paper/tools/build.py           # pdflatex, bibtex, pdflatex x2; CPU ledger in build_log.jsonl
```

`build.py` gives every child process `RLIMIT_CPU` equal to the remaining budget of 600 CPU-s and
appends each step to `build_log.jsonl`, which also records the `pdftoppm` render checks. Its
`build_report.json` records the page count, the page on which the main text ends (label
`sec:end-of-main-text`) and the warning counts.

The build status (engine, version, pages, warnings) is recorded in `PAPER_STATUS.md` after each
build. The build-and-check CPU budget is 600 s. It is tracked separately from the analysis budget.
