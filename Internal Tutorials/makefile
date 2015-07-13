TARGET=Main

.PHONY: clean
# all

all:
	pdflatex ${TARGET}
	bibtex ${TARGET}
	pdflatex ${TARGET}
	pdflatex ${TARGET}

clean:
	rm -rf *.aux *.dvi *.lof *.log *.toc *.lot *.ps *.bbl *.blg *.backup *.out *.tex~
	rm -rf struct/*.aux
