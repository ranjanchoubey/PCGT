# PCGT Paper — LaTeX Source

## Compiling

From the `paper/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use VS Code + LaTeX Workshop (auto-compiles on save).

## Structure

```
paper/
├── main.tex              # Main document
├── references.bib        # Bibliography
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_method.tex
│   ├── 04_experiments.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   ├── 07_limitations.tex
│   └── 09_appendix.tex
└── figures/
    ├── convergence.pdf
    └── runtime_comparison.pdf
```
