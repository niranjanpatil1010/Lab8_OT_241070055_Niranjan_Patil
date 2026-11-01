# Track E prompt log (tool: Claude, claude.ai chat, Claude Sonnet 5.5)

*The student wrote the original prompts in Hinglish; they are given here translated into standard English.*

## Original prompt (with the assignment manual pasted)
"Choose a problem for Track E that is easy to implement, shows a measurable improvement, and connects directly to the algorithms of the Optimization Techniques course."

## AI response (excerpt)
Recommended **exam / lab-batch timetabling (clash minimization)**: binary variables x[e,t,r], objective w1·slots + w2·back-to-back + w3·unused seats, hard constraints (one slot and room per exam, room capacity, no student clash); methods: random baseline, greedy, ILP, Branch-and-Bound; instance sizes 20 to 200.

## Revised prompts
1. "Will the explanation above be too hard?" -> the AI offered a simplified version (no rooms, a single objective).
2. "Don't make it that easy; make it a reasonably good version and finalize this one." -> the AI restored rooms, the 3-term objective, four methods plus greedy + local search, and size/sensitivity experiments (final scope in REPORT.md, Track E).
3. "Give me the complete lab project: report and everything the manual requires." -> the AI built the code in `trackE/` (data.py, model.py, methods.py, run.py) and the tests.

## Verification note
Every method's output is checked by an independent `validate()`; ILP, greedy and random outputs are asserted feasible; the hand-written B&B was cross-checked against the ILP on a 4-exam instance (this caught the invalid symmetry pruning, FAILURE_LOG #1 and #3). Results are in `results/trackE_results.json`; report numbers are copied from that file; the ILP did not reach optimality (stated in the report).

## Commit
f846a4c  [AI:claude] Track E: exam timetabling (random, greedy, greedy+LS, ILP, B&B) + tests
80f49f7  [AI:claude] Report sections A/E, checklist, failure log
