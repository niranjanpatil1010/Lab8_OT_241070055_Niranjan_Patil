# Run guide — how to run everything and which screenshots to take

## 0. Setup (once)
Python 3.10+ (tested on Python 3.12, Linux; also run on Windows).
```
pip install numpy scipy matplotlib pytest pennylane scikit-learn coverage
```
Run every command from the repository root. Each track has its own folder; always `cd` into it as shown (some scripts write to `../results/`).

## 1. Run each track
| Track | Command | What you should see | Screenshot? |
|---|---|---|---|
| A Continuous | `cd trackA && python3 run.py` | optimizer results, backprop gradient-check error ≈ 1e-9, saddle/local-minima output | ✅ |
| B Discrete | `cd trackB && python3 run.py` | knapsack/scheduling: greedy vs DP vs B&B vs ILP; Max-Cut QUBO/Ising | ✅ |
| C Quantum optimization | `cd trackC && python3 run.py` | QAOA on the Track B Max-Cut, VQE, parameter-shift/SPSA/QNG, shots, noise, barren plateaus (simulated). Needs `results/trackB_results.json` (already included) | ✅ |
| D Submodular / parallel / distributed | `cd trackD && python3 run.py` then `python3 run_locality_sync.py` | submodularity check, load balancing, speedup, FedAvg, MGDA/GradNorm; locality and sync/communication timings | ✅ |
| E Exam timetabling | `cd trackE && python3 run.py 20 40 60 100` | table: random / greedy / greedy+LS / ILP (about 5 minutes: ILP has a 60 s limit per size) | ✅ |
| F Compiler | `cd trackF && python3 run.py` | 5 optimization passes, instruction counts, identical program output, register-allocation greedy vs ILP | ✅ |
| G xv6 | results are saved: `python3 -c "import json;print(json.dumps(json.load(open('results/trackG_results.json'))['aggregate'],indent=1))"` | lottery vs round-robin CPU shares, ping-pong, fairness. Re-running needs QEMU + a RISC-V toolchain: see `trackG/README_G.md` | ✅ (saved results + `trackG/xv6_patch.diff`) |
| H DBMS | `cd trackH && python3 run.py` | 100k-row SQLite, index selection greedy vs DP vs ILP, EXPLAIN plans, identical results | ✅ |
| I Quantum algorithms / QML | `cd trackI && python3 run.py` | DJ, BV, Grover, QFT, QPE, Shor, annealing, quantum kernel vs RBF SVM | ✅ |
| J Test-suite minimization | `python3 trackJ/profile_suite.py` then `python3 trackJ/run.py` | real per-test timings/coverage, ILP vs greedy, re-measured time and coverage | ✅ |
| Tool comparison | `cd compare && python3 run_comparison.py` | Antigravity vs Claude on Track B against brute force | ✅ |

## 2. Run all tests
```
python3 -m pytest tests -q
```
Expected: `88 passed, 1 skipped` (the skipped Track G test needs a pristine xv6 clone; set `XV6_PRISTINE` to run it). **Take a screenshot of this.**

## 3. Screenshots to put in `screenshots/`
Name them by track, e.g. `trackA.png`, `trackE_table.png`, `tests_passed.png`. Capture the terminal with the command visible at the top and the result below. Also useful: `results/trackE_plot.png`, `results/trackD_locality_plot.png`, `results/trackG_shares_plot.png` (already in the repo).

## 4. Publish to GitHub
1. On github.com create a new **public** repository (empty: no README, no license).
2. In the unzipped project folder (it already contains the git history):
```
git status                       # should say: nothing to commit
git add screenshots && git commit -m "[manual] add terminal screenshots"
git remote add origin https://github.com/<your-username>/optimize-anything.git
git branch -M main
git push -u origin main
```
3. If Git asks for a password, use a Personal Access Token (GitHub → Settings → Developer settings → Personal access tokens), or log in with `gh auth login` / GitHub Desktop.
4. If Git complains about identity: `git config user.name "Your Name"` and `git config user.email "you@example.com"`.
5. Open the repository link in a private/incognito window to confirm it is public and the README shows.

## 5. Before you submit — checklist
- [ ] `python3 -m pytest tests -q` passes
- [ ] screenshots added to `screenshots/` and committed
- [ ] README → AI tools register: edit it if you know the exact model Antigravity used for each track
- [ ] repository is public; link submitted together with `REPORT.md`
