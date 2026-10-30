# Track F Prompt Log

## Original prompt (excerpt)
```
Track F — Compiler optimization:
mini-compiler (lexer, parser, IR, codegen) for a toy language with >=4 optimizations
(constant folding/propagation, dead-code elimination, CSE, strength reduction, loop-invariant motion...).
Pose one decision (pass order, register allocation or unroll factor) as an optimization and solve with greedy/ILP/DP.
Measure instruction count, runtime, code size, compile time;
optimized and unoptimized programs must give identical output.
```

## AI Response / Implementation Approach
- Implemented in `trackF/compiler.py` and `trackF/run.py`:
  - Lexer: Regex tokenizer for ToyLang.
  - Parser: Recursive descent AST parser.
  - IR: 3-Address Code (3AC) quadruples.
  - Virtual Machine interpreter with execution cycle profiling and output interception.
  - 5 Optimization Passes:
    1. Constant Folding & Constant Propagation.
    2. Common Subexpression Elimination (CSE).
    3. Strength Reduction ($x \times 2 \to x + x, x \times 0 \to 0, x \times 1 \to x$).
    4. Dead Code Elimination (DCE).
    5. Loop-Invariant Code Motion (LICM).
  - Decision Optimization: Register Allocation via Graph Coloring formulated as Integer Linear Programming (ILP) using `scipy.optimize.milp` vs Greedy (Chaitin-Briggs / Welsh-Powell).
  - Measured: Code size, dynamic VM instruction cycles, runtime, compile time, and register spills across multiple benchmarks.

## Verification Note
- 8 unit tests in `tests/test_trackF.py` all pass.
- Verified terminal output identity (`assert out_unopt == out_opt`) on all programs.
- ILP eliminated 22 spills (30.14% reduction) over greedy register allocation across 20 test topologies.

## Bugs Found & Logged (see FAILURE_LOG.md #11)
- VM interpreter `PRINT` evaluated constant immediate operands as variable environment keys, caught by the identity validator.
