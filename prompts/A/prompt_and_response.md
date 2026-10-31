# Track A prompt log (tool: Claude, claude.ai chat, Claude Sonnet 5.5)

## Original prompt (student)
"…muje khud kuch karna nahi he okk sab tum karo and ha iska permission diya he sir ne" (after the assignment manual was pasted; AI proposed building track by track and announced "Abhi Track A (continuous optimization) kar raha hoon").

## AI response (excerpt)
`trackA/run.py`, numpy only, seeded: GD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton on a convex quadratic and Rosenbrock (same 9-value lr grid); noisy linear regression (GD vs SGD vs mini-batch, 10 seeds); Gauss–Newton vs GD; hand-written backprop MLP with L2 and a finite-difference gradient check; Hessian-vector products, Fisher diagonal, power iteration; saddle and local-minima landscapes.

## Revised prompt / correction
No prompt change; the AI reviewed its own output before reporting and replaced an untested hard-coded claim ("Newton goes to the saddle") by a real run (FAILURE_LOG #2).

## Verification note
Gradient check max relative error 2.1e-9; HVP vs exact Hessian 9.4e-12; power-iteration eigenvalue equals exact; all values in REPORT.md Track A copied from `results/trackA_results.json`; `tests/test_trackA.py` validates the saved results.

## Commit
06499b2  [AI:claude] Track A: continuous optimizers, backprop check, curvature, landscapes
