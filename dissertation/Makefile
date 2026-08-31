.PHONY: default
default: diss.pdf

%.pdf: %.tex
	pdflatex $<   # generates .aux file used by bibtex
	bibtex $(basename $<) # generates actual bibliography (.bbl file)
	pdflatex $<   # updates citations
	pdflatex $<   # updates cross-references
