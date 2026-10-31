# Track G prompt log
## Original prompt (excerpt, from the user's task file)
"G: xv6 under QEMU: implement one kernel feature (e.g. MLFQ/lottery/stride scheduler or copy-on-write fork), benchmark on >=3 workloads with new user-level programs, xv6 must boot and pass usertests; save the patch as a diff. If QEMU is missing: write patch + benchmarks + run instructions and mark results NOT YET MEASURED — do not invent numbers."
## AI response / approach (Claude Sonnet 5.5 via Claude Code, cloud sandbox with QEMU 8.2.2 installed)
- QEMU and riscv64 toolchain turned out to be installable, so results are MEASURED (the "NOT YET MEASURED" fallback was not needed).
- Feature: lottery scheduler (`SCHED=lottery` build flag) + syscalls `settickets`, `getpinfo`, `schedseed`; per-process tick accounting (`charge_tick` in timer trap path); baseline = original round-robin with identical accounting.
- Benchmarks: `user/schedbench.c` (prop, pingpong, fair, solo, forkstorm); driver `trackG/run.py`, `trackG/xv6run.py`.
## Revised prompts / corrections
1. copyout signature mismatch (FAILURE_LOG #14). 2. Benchmark start-up starvation (#15). 3. fs.img write lock (#16).
## Verification note
usertests: logs in trackG/logs/. Patch checked with `git apply --check` on pristine commit e93dabe. Unit tests: tests/test_trackG.py. Raw QEMU output: results/trackG_raw.txt.
## Commit
Not committed by the assistant (no commit was requested); user commits with prefix per README.
