"""Correctness tests: validator catches violations; B&B (own code) matches ILP on a tiny instance."""
import sys; sys.path.insert(0, "../trackE")
from data import generate
from model import objective, validate
import methods as M

def test_validator_catches_clash():
    inst = generate(10, seed=1)
    bad = {e: (0, 5) for e in range(10)}          # everything in one slot+room
    assert validate(inst, bad, 5)

def test_greedy_feasible():
    inst = generate(30, seed=2)
    g = M.greedy(inst); T = max(t for t, _ in g.values()) + 1
    assert not validate(inst, g, T)

def test_bnb_equals_ilp_tiny():
    inst = generate(4, seed=3)
    g = M.greedy(inst); T = max(t for t, _ in g.values()) + 2
    ub = objective(inst, g)[0]
    bb, info = M.branch_and_bound(inst, T, ub)
    il, _ = M.ilp(inst, T, 30)
    assert not validate(inst, bb, T)
    assert abs(objective(inst, bb)[0] - objective(inst, il)[0]) < 1e-6
    print("B&B nodes:", info["nodes"], "obj:", objective(inst, bb)[0])
