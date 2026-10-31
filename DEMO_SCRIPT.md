# 5-minute demo script (screen-record the terminal + REPORT.md)
1. (30 s) Problem: 10 tracks; main case study = exam timetabling. Show REPORT.md Track E formulation.
2. (60 s) `cd trackE && python3 run.py 20 40` — show the table: random vs greedy vs greedy+LS vs ILP. Say: greedy+LS best, ILP hit the 60 s limit with a gap (honest negative result).
3. (60 s) `python3 -m pytest tests -q` — show all passed. Mention the independent `validate()` and the B&B-vs-ILP check that caught a real bug (FAILURE_LOG #1).
4. (60 s) Pick two highlights from REPORT.md: Track G lottery scheduler (CPU share 51.6/34.5/13.9 % vs tickets 30/20/10) and Track I Grover (query advantage but slower wall-clock on a simulator).
5. (60 s) Lessons: ILP not always best; parallel speedup < 1 when tasks are tiny (D2, D6); locality: column access 2.9× slower (D5); AI made 19 logged mistakes; tool comparison found no difference on Track B.
6. (30 s) Show README AI-tools register and the failure log.
