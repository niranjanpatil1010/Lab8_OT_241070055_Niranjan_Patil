"""Objective + independent validator (used to check EVERY method's output)."""
from data import W1, W2, W3


def objective(inst, sol):
    """sol: dict exam -> (slot, room). Returns (total, (slots_used, back_to_back, unused_seats))."""
    n, C, size, caps = inst["n"], inst["C"], inst["size"], inst["caps"]
    slots_used = len({t for t, _ in sol.values()})
    b2b = 0
    for e in range(n):
        for f in range(e + 1, n):
            if C[e][f] and abs(sol[e][0] - sol[f][0]) == 1:
                b2b += C[e][f]
    unused = sum(caps[r] - size[e] for e, (t, r) in sol.items())
    return W1 * slots_used + W2 * b2b + W3 * unused, (slots_used, b2b, unused)


def validate(inst, sol, T):
    """Return list of hard-constraint violations (empty list == feasible)."""
    n, C, size, caps = inst["n"], inst["C"], inst["size"], inst["caps"]
    errs = []
    if set(sol) != set(range(n)):
        return ["not every exam assigned exactly once"]
    seen = {}
    for e, (t, r) in sol.items():
        if not (0 <= t < T):
            errs.append(f"exam {e}: slot {t} out of range")
        if caps[r] < size[e]:
            errs.append(f"exam {e}: room {r} too small")
        if (t, r) in seen:
            errs.append(f"room clash: exams {seen[(t, r)]} & {e}")
        seen[(t, r)] = e
    for e in range(n):
        for f in range(e + 1, n):
            if C[e][f] and sol[e][0] == sol[f][0]:
                errs.append(f"student clash: exams {e} & {f} share slot {sol[e][0]}")
    return errs
