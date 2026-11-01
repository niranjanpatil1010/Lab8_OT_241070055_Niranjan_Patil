# Optimize Anything — Assignment 8 (Optimization Techniques, VJTI)

Status: **All ten tracks A–J implemented, run and written up** (see CHECKLIST.md, REPORT.md). Terminal screenshots of the runs go in `screenshots/` (see RUN_GUIDE.md).

## Setup
```
pip install numpy scipy matplotlib pytest pennylane scikit-learn coverage
cd trackE && python3 run.py 20 40 60 100     # writes results/trackE_results.json (same pattern for every track: python3 trackX/run.py)
cd .. && python3 -m pytest tests -q          # all 84 tests in ONE session (needs trackJ profile: python3 trackJ/profile_suite.py)
# Track J:  python3 trackJ/profile_suite.py && python3 trackJ/run.py
# Track G:  needs qemu-system-riscv64 + riscv64 gcc; see trackG/README_G.md
```
Hardware/settings: single CPU, HiGHS via scipy.optimize.milp, ILP time limit 60 s, seeds fixed (instance seed 0, random baseline seeds 0-9).

## AI tools register (every tool used)
| Tool + model version | Tracks | Dates | Kind of work |
|---|---|---|---|
| Claude (claude.ai chat) / Claude Sonnet 5.5 | A, E; D5/D6 (locality, sync/communication); tool comparison; prompt logs, checklist, packaging | Oct 2026 | planning, code generation, debugging, testing, documentation |
| Antigravity / Claude Sonnet 4.6 Thinking & Gemini 3.8 Flash (the log for Track B names Sonnet 4.6 Thinking; the exact model per track was not recorded) | B, C, D, F, H, I, J (first draft) | 2026-10-28 | planning, code generation, debugging, tests, report |
| Claude Code / Claude Sonnet 5.5 | G, J (rewrite), tests/conftest.py, tests/test_trackA.py, final questions | 2026-10-28/29 | code generation, debugging, testing, documentation |

## Commit convention and honesty note
`[AI:claude]` = claude.ai chat, `[AI:antigravity]`, `[AI:claude-code]`, `[manual]` = the student's own edits. **The git history was created in one session after the work was done**, grouping files by the tool that generated them (taken from the chat transcripts); it does not show the original minute-by-minute timeline. Track A and E commits (`06499b2`, `f846a4c`, `80f49f7`) were made while those tracks were built. The raw chat exports in `prompts/exports/` are the primary evidence.

## Layout
`trackA..J/` code (`trackG/xv6_patch.diff` = kernel patch) · `compare/` tool comparison · `tests/` · `results/` · `prompts/A..J/` + `prompts/exports/` · `REPORT.md` · `CHECKLIST.md` · `FAILURE_LOG.md` · `RUN_GUIDE.md` · `screenshots/`

## Tool comparison (manual §8): Antigravity vs Claude (claude.ai chat) on Track B
Task: 0/1 knapsack with greedy, DP, branch-and-bound and ILP; Max-Cut as QUBO and Ising. Antigravity's code is `trackB/run.py`; Claude's independent re-write is `compare/claude_trackB_alt.py`. Both were run by `compare/run_comparison.py` against brute-force truth (40 random knapsack instances, n=15; 20 random 8-node graphs). Raw: `results/tool_comparison.json`.

| Measure | Antigravity | Claude (chat) |
|---|---|---|
| Knapsack DP / B&B / ILP optimal | 40/40 each | 40/40 each |
| Knapsack greedy optimal | 25/40 | 25/40 |
| Infeasible answers | 0 | 0 |
| Mean time per instance (greedy / DP / B&B / ILP) | 0.16 / 0.10 / 0.12 / 8.1 ms | 0.09 / 0.11 / 0.12 / 8.3 ms |
| Max-Cut: QUBO minimum = true max cut | 20/20 | 20/20 |
| QUBO→Ising energy identity exact | 20/20 | 20/20 |
| Hallucinations found by the validators | none in the final code | none |
| Prompt iterations to a passing state | ~4 fix rounds in its own log (zero-edge graph, encoding, import-time `json.dump`, two wrong hand-computed test values; FAILURE_LOG #5, #6) | solver correct on the first run; my *comparison harness* needed 3 fixes (FAILURE_LOG #18) |

Conclusion: no measurable difference in correctness or speed on this task. **Caveats:** one small task; the tools were not given the identical prompt text (Antigravity got the master prompt, Claude a paraphrase of the manual); Claude's run was **not blind** — I had read Antigravity's function signatures before writing my version; generation time was not measured; my file has two cosmetic dead lines in `qubo_to_ising`/`maxcut_qubo`.
