# Track A prompt log (tool: Claude, claude.ai chat, Claude Sonnet 5.5)

*The student wrote the prompts in Hinglish; they are given here translated into standard English.*

## Original prompt
"Build the whole lab project for me: the report and everything the manual requires." The AI proposed building it track by track and started with Track A (continuous optimization).

## AI response (excerpt)
`trackA/run.py`, numpy only, seeded: GD, Momentum, NAG, AdaGrad, RMSProp, Adam and Newton on a convex quadratic and on Rosenbrock (same 9-value learning-rate grid); noisy linear regression (GD vs SGD vs mini-batch, 10 seeds); Gauss-Newton vs GD; a hand-written backpropagation MLP with L2 regularization and a finite-difference gradient check; Hessian-vector products, Fisher diagonal and power iteration; saddle-point and local-minima landscapes.

## Revised prompt / correction
No prompt change. The AI reviewed its own output before reporting and replaced an untested hard-coded claim ("Newton goes to the saddle") with a real run (FAILURE_LOG #2).

## Verification note
Gradient check: max relative error 2.1e-9. HVP vs exact Hessian: 9.4e-12. Power-iteration eigenvalue equals the exact eigenvalue. All Track A numbers in REPORT.md are copied from `results/trackA_results.json`; `tests/test_trackA.py` validates the saved results.

## Commit
06499b2  [AI:claude] Track A: continuous optimizers, backprop check, curvature, landscapes
