CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic

.PHONY: all analysis manuscript clean

all: analysis manuscript

analysis: Fig1.pdf Fig2_revised.pdf Fig3_revised.pdf

revision_analysis: revision_analysis.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

revision_stochastic_results.csv: revision_analysis
	./revision_analysis $@

Fig1.pdf deterministic_results.npz: deterministic_analysis.py
	python deterministic_analysis.py

Fig2_revised.pdf Fig3_revised.pdf: revision_figures.py revision_stochastic_results.csv
	python revision_figures.py

manuscript: main.pdf response_to_reviewers.pdf

main.pdf: main.tex references.bib sn-jnl.cls sn-mathphys-ay.bst analysis
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

response_to_reviewers.pdf: response_to_reviewers.tex
	pdflatex -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
	pdflatex -interaction=nonstopmode -halt-on-error response_to_reviewers.tex

clean:
	rm -f revision_analysis *.aux *.bbl *.blg *.log *.out *.toc
