DOCNAME=report

all: report

.PHONY: clean

report:
	pdflatex $(DOCNAME).tex
	biber $(DOCNAME).bcf
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex

view: report
	open $(DOCNAME).pdf

clean:
	rm *.blg *.bbl *.aux *.log
