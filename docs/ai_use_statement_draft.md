# AI Use Statement — draft for author review

**Status: draft. Not inserted into the manuscript.** The bracketed passages are
attestations only an author can make. Replace or delete each one; do not sign
anything you have not checked yourself.

This draft errs toward disclosing more than may be required. Over-disclosure
costs nothing; under-disclosure is a research-integrity problem.

---

## Draft text

> **AI Use Statement**
>
> Language models are the object of study in this work. Qwen2.5-32B-Instruct-AWQ
> and Mistral-Nemo-Instruct act as the ReAct agents whose failed trajectories are
> repaired, and a same-model prompt supplies the diagnosis/replay comparator.
> Their versions, revisions and decoding settings are pinned in each frozen
> protocol.
>
> Generative AI assistance was also used extensively in producing this work, and
> the authors wish to be specific about its scope. An AI coding assistant
> contributed to: the design of the length-matched swap control and its exact
> sign-flip analysis; implementation of the experiment runners, auditors and
> paper-asset builders; orchestration of the cloud experiments, including budget
> gating and result retrieval; the statistical diagnostics reported in the
> appendices; drafting and revision of manuscript text; and the test suite that
> checks the above. Literature lookup and artifact auditing were likewise
> assisted.
>
> The following safeguards are structural rather than discretionary. Every
> numerical claim in this paper is generated from audited records by a build
> step, not transcribed by hand; the manuscript fails to build if a cited
> quantity has no generating source. Each study is frozen by checksum before
> execution and re-analysed independently afterwards, and every reported cell
> reproduces with zero mismatches between the on-instance and local analyses. A
> supplement reproduces all four primary test families from an isolated
> extraction.
>
> These safeguards constrain arithmetic and provenance. They do not establish
> that the scientific reasoning, the choice of estimands, or the interpretation
> of null results is correct. [AUTHORS: state here what you verified
> independently, and by what means. Be specific — for example, which analyses
> you re-derived yourself, which code you read line by line, and which claims
> you checked against the raw records rather than against generated summaries.]
>
> [AUTHORS: state here who reviewed the work, what each reviewer checked, and
> what remained unverified at submission.]
>
> The authors take full responsibility for all content, including any errors
> introduced by AI assistance.

---

## Why the draft says what it says

**It names the assistance broadly.** The previous placeholder listed "manuscript
revision, code inspection, tests, literature lookup and experiment
orchestration, artifact auditing and interpretation of results." That
understates it: the swap control's design, its analysis method, and the
appendix sections reporting it were all AI-assisted. A reader comparing the
statement to the commit history should not find a gap.

**It separates what the safeguards do prove from what they do not.** Generated
numbers and frozen protocols give strong guarantees about arithmetic and
provenance. They give none about whether the design answers the question, or
whether a null result is being framed fairly. Conflating the two would be the
easiest way for this statement to mislead.

**It leaves the verification claims blank.** The statement is worthless if it
asserts checking that did not happen. Those brackets are the point of the
draft, not an oversight.

## Known facts you may want to reflect

Errors that reached execution during AI-assisted work on this project, all
caught and fixed, offered here because they bear on how much independent
checking is warranted:

- A budget gate was wired into two of three call sites, so a controller refused
  two minutes into a paid GPU session.
- A prediction that more matched pairs would lower the resolution floor was
  wrong; the tie rate binds instead.
- Editing a script invalidated the checksum provenance of an already-completed
  study package.
- Three figures shipped with labels obscured by legends, caught only by
  inspecting the rendered images.
- Tests written against one pandas version failed on the instance's version,
  aborting a run at preflight.

None of these reached a reported result. All were found by continued checking
rather than by a process that guarantees detection.
