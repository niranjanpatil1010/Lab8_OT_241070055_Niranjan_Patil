"""Synthetic exam-timetabling instance generator (seeded -> reproducible)."""
import numpy as np

ROOM_CAPS = [20, 30, 40, 60, 80, 120]   # R = 6 rooms available in every slot
W1, W2, W3 = 10.0, 1.0, 0.1             # weights: slots used, back-to-back, unused seats


def generate(n_exams, seed=0, group_size=10, per_student=5):
    """Students belong to a branch (group of ~10 exams): 4 exams from it + 1 elective
    from anywhere -> realistic, sparse conflict graph."""
    rng = np.random.default_rng(seed)
    n_students = n_exams * 3
    n_groups = max(1, n_exams // group_size)
    group_of = np.arange(n_exams) % n_groups
    enrol = []
    for _ in range(n_students):
        g = rng.integers(n_groups)
        pool = np.where(group_of == g)[0]
        k = min(per_student - 1, len(pool))
        own = rng.choice(pool, size=k, replace=False)
        extra = rng.integers(n_exams)
        enrol.append(sorted(set(own.tolist() + [int(extra)])))
    size = np.zeros(n_exams, dtype=int)
    C = np.zeros((n_exams, n_exams), dtype=int)   # C[e][f] = students taking both e and f
    for s in enrol:
        for e in s:
            size[e] += 1
        for i in s:
            for j in s:
                if i != j:
                    C[i][j] += 1
    assert size.max() <= max(ROOM_CAPS), "an exam is larger than the biggest room"
    return dict(n=n_exams, size=size, C=C, caps=list(ROOM_CAPS), seed=seed)
