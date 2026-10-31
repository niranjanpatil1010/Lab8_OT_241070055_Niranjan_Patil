# Track E prompt log (tool: Claude, claude.ai chat, Claude Sonnet 5.5)

## Original prompt (student, Hinglish, with the assignment manual pasted)
"give this to me problem choose karna chahiye jo implement karna easy ho, measurable improvement dikhaye, aur Optimization Techniques ke algorithms se directly connect ho."

## AI response (excerpt)
Recommended **exam/lab-batch timetabling (clash minimization)**: binary x[e,t,r], objective w1·slots + w2·back-to-back + w3·unused seats, hard constraints (one slot+room per exam, capacity, no student clash); methods random baseline, greedy, ILP, Branch-and-Bound; sizes 20→200.

## Revised prompts
1. "uper vala explaination me hard nahi na jayega" → AI offered a simpler version (no rooms, one objective).
2. "nahi nahi itana bhi easy mat kar achha kahsa kar isko hi okk kar ab" → AI restored rooms, 3-term objective, 4 methods + greedy+local search, sensitivity/size experiments (final scope in REPORT.md Track E).
3. "…muje final de lab ka sab project report and all jo manual me chahiye o sab okk" → AI built the code in `trackE/` (data.py, model.py, methods.py, run.py) and tests.

## Verification note
Every method's output is checked by an independent `validate()`; ILP/greedy/random all asserted feasible; own B&B cross-checked against ILP on a 4-exam instance (caught the invalid symmetry pruning, FAILURE_LOG #1, #3). Results in `results/trackE_results.json`; report numbers copied from that file; ILP did not reach optimality (stated in REPORT).

## Commit
f846a4c  [AI:claude] Track E: exam timetabling (random, greedy, greedy+LS, ILP, B&B) + tests
80f49f7  [AI:claude] Report sections A/E, checklist, failure log (4 real cases)
