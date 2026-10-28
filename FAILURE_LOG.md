# Failure log (manual needs >= 3; only add cases that REALLY happened to you)

## #1 Invalid symmetry-breaking in hand-written Branch-and-Bound (Track E) — REAL
- What: B&B only allowed opening "the next new slot" (slots interchangeable assumption). On a 8-exam instance it returned 136.2 while the ILP returned 121.2, so B&B was not exact.
- Caught by: `tests/test_trackE.py::test_bnb_equals_ilp_tiny` (cross-check B&B vs ILP).
- Cause: the back-to-back term depends on slot ORDER, so slots are not symmetric. (The ILP's `y[t] >= y[t+1]` rule is valid because it only forces used slots to be a prefix, which is a shift, not a permutation.)
- Fix: removed the pruning; B&B now explores every slot. Cost: pure-Python B&B only scales to ~4-5 exams (140,935 nodes for 4 exams), so tests use n=4.
- Lesson: symmetry-breaking must be checked against the objective, not just the constraints.

## #2 Untested claim hard-coded in my own results (Track A) — REAL
- What: the first version of `trackA/run.py` stored a text note "Newton is attracted to saddles… goes to (0,0)" without running anything to check it.
- Caught by: review of the output before writing the report (the field held a string, not a measurement).
- Fix: added `newton_sad()`; real run from (0.3,0.5) converges to (0,0) — claim now measured.
- Lesson: report only what an experiment produced (manual §10).

## #3 Exact B&B did not scale — inefficiency — REAL
- What: after removing the invalid pruning (#1) the pure-Python B&B ran >5 minutes on 6-8 exams and hit the tool time limit.
- Fix: B&B test restricted to a 4-exam instance (140,935 nodes, 2.6 s); documented as non-scalable.
- Lesson: weak lower bound (partial cost only) -> use solver B&B (HiGHS) for real sizes.

## #4 Shell command written for the wrong shell — REAL (tooling)
- What: AI-generated `mkdir -p dir/{a,b}` ran in `/bin/sh`, which has no brace expansion; it created literal wrong directories and the following `cd` failed, so no files were written.
- Fix: re-ran with `bash -c`. Lesson: check which shell executes generated commands.
