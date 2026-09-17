.PHONY: iclr-draft submission

iclr-draft:
	python ../scripts/build_iclr_draft.py
	python ../scripts/build_followup_paper.py
	python ../scripts/build_runtime_paper.py
	python ../scripts/build_extension_paper.py
	python ../scripts/position_pair_paper.py
	python ../scripts/replication_diagnostics.py
	python ../scripts/outcome_sensitivity.py
	python ../scripts/runtime_scope.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/iclr2027 iclr2027.tex
	mkdir -p ../output/pdf
	cp build/iclr2027/iclr2027.pdf ../output/pdf/agent-repair-iclr2027-draft.pdf

# A submission-labelled build. The working copy is deliberately stamped
# "Working draft; not submitted" so an unsubmitted PDF can never pass for a
# submission; this target flips \submissionfalse to \submissiontrue and
# restores the official ICLR labels. Use it for an actual submission and for
# the LLM review service. The generated .tex is an artifact, not a source.
submission: iclr-draft
	sed 's/^\\submissionfalse$$/\\submissiontrue/' iclr2027.tex > iclr2027_submission.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -outdir=build/submission iclr2027_submission.tex
	mkdir -p ../output/pdf
	cp build/submission/iclr2027_submission.pdf \
	  ../output/pdf/agent-repair-iclr2027-submission.pdf
	rm -f iclr2027_submission.tex
	@pdftotext -layout ../output/pdf/agent-repair-iclr2027-submission.pdf - \
	  | grep -q 'Under review as a conference paper at ICLR 2027' \
	  && echo "submission build OK: official header present" \
	  || (echo "REFUSING: submission PDF lacks the ICLR header" && exit 1)
