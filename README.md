# Defector-responsive strategy switching: revised analysis

This repository reproduces the revision of **Analytical and Finite-Population Properties of Defector-Responsive Strategy Switching** submitted to *Dynamic Games and Applications*.

## What changed in the revision

The finite-population model is now a continuous-time Moran-type process with two independent event channels:

1. Birth-death events, with payoff-dependent individual birth rates and uniform replacement.
2. Direct ALLC-to-TFT phenotype transitions, at constant intensity or intensity proportional to current ALLD frequency.

The C++ simulation implements the exact event sequence of this continuous-time process. The manuscript derives its generator and shows that its large-population drift matches the deterministic replicator-switching equations at selection strength `w = 1`.

The revised numerical study includes:

- Post-rescue composition at `10N, 20N, ..., 100N` birth-death events.
- Population sizes `N = 50, 100, 200`.
- Selection strengths `w = 0.25, 0.5, 1`.
- Benefit-to-cost ratios `b/c = 3, 5, 7`.
- Readiness costs `k = 0, 0.2, ..., 2`.

## Files

- `main.tex`, `references.bib`: revised manuscript source.
- `main.pdf`: compiled clean manuscript.
- `response_to_reviewers.tex`, `response_to_reviewers.pdf`: point-by-point response.
- `deterministic_analysis.py`: deterministic integrations and Figure 1.
- `revision_analysis.cpp`: continuous-time stochastic simulations.
- `revision_figures.py`: Figures 2 and 3 plus validation checks.
- `revision_stochastic_results.csv`: complete stochastic summary data.
- `deterministic_results.npz`: deterministic numerical data.
- `Fig1.pdf`, `Fig2_revised.pdf`, `Fig3_revised.pdf`: vector figures used in the manuscript.
- `sn-jnl.cls`, `sn-mathphys-ay.bst`: Springer Nature template files.

## Reproduce everything

Requirements:

- Python 3.10 or later
- A C++17 compiler such as `g++`
- A LaTeX installation with `pdflatex` and `bibtex`

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete workflow:

```bash
make all
```

The stochastic simulation uses fixed seeds. On completion, the plotting script checks that no trial was censored, the two zero-switching controls match, and the post-rescue preservation advantage remains positive in all 27 robustness conditions.

For a fast implementation check, compile the C++ program and pass `--quick` as the second argument:

```bash
g++ -O3 -std=c++17 revision_analysis.cpp -o revision_analysis
./revision_analysis quick_results.csv --quick
```

## Model scope

The proportional switching rule was introduced by Pal, Lambert, and Nowak (2025), Supplementary Figure S8. This work does not claim the rule as new. It studies its cost threshold, constructs a finite process with a direct deterministic limit, and tests its finite-population tradeoff across parameters and horizons.

## License and citation

Please cite the manuscript and the public preprint associated with this repository. The code is provided for research reproducibility.
