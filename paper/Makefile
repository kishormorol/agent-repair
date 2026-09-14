.PHONY: iclr-draft

iclr-draft:
	python ../scripts/build_iclr_draft.py
	python ../scripts/build_followup_paper.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/iclr2027 iclr2027.tex
	mkdir -p ../output/pdf
	cp build/iclr2027/iclr2027.pdf ../output/pdf/agent-repair-iclr2027-draft.pdf
