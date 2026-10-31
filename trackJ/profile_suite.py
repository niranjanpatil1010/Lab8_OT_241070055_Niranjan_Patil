"""Track J step 1 (MEASURED): profile the repo's own pytest suite.
 - per-test wall time: pytest junit XML, median of REPEATS plain runs (no coverage)
 - per-test coverage: coverage.py with dynamic_context=test_function; requirement = source function (file::def)
 - codebase size: non-blank, non-comment source lines in track*/ (tests excluded)
Writes results/trackJ_profile.json. Excludes tests/test_trackJ.py (it imports this track's own output).
Run from repo root:  python3 trackJ/profile_suite.py [repeats]"""
import os, sys, ast, json, glob, subprocess, statistics, tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IGN = ["--ignore=tests/test_trackJ.py", "-p", "no:cacheprovider", "-q"]

def loc():
    tot, files = 0, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "track*", "*.py"))):
        n = sum(1 for l in open(f, encoding="utf8") if l.strip() and not l.strip().startswith("#"))
        files[os.path.relpath(f, ROOT)] = n; tot += n
    return tot, files

def timed_run(repeats):
    per = {}
    for _ in range(repeats):
        x = tempfile.mktemp(suffix=".xml")
        subprocess.run([sys.executable, "-m", "pytest", "tests", f"--junitxml={x}"] + IGN, cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        for tc in ET.parse(x).getroot().iter("testcase"):
            per.setdefault((tc.get("classname"), tc.get("name")), []).append(float(tc.get("time")))
        os.unlink(x)
    return {k: statistics.median(v) for k, v in per.items()}, per

def nodeid(classname, name):
    parts = classname.split(".")
    i = next(k for k, p in enumerate(parts) if p.startswith("test_"))
    return "/".join(parts[:i + 1]) + ".py" + "".join("::" + p for p in parts[i + 1:]) + "::" + name

def covered_functions(node_ids):
    """Independent validator: run ONLY node_ids under coverage.py, return the set of covered file::func."""
    import coverage
    d = tempfile.mkdtemp(); dat = os.path.join(d, ".coverage"); rc = os.path.join(d, "rc")
    open(rc, "w").write(f"[run]\ndata_file = {dat}\nsource = {ROOT}\nomit = */tests/*\n")
    subprocess.run([sys.executable, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest"] + node_ids + IGN,
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    cd = coverage.CoverageData(basename=dat); cd.read(); out = set()
    for f in cd.measured_files():
        rel = os.path.relpath(f, ROOT)
        if not (rel.startswith("track") and rel.endswith(".py")): continue
        fr = func_ranges(f)
        for line in cd.lines(f) or []:
            names = [n for (a, b, n) in fr if a < line <= b]
            if names: out.add(f"{rel}::{names[-1]}")
    return out

def func_ranges(path):
    t = ast.parse(open(path, encoding="utf8").read()); out = []
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((n.lineno, n.end_lineno, n.name))
    return out

def coverage_matrix():
    import coverage
    dat = os.path.join(tempfile.mkdtemp(), ".coverage")
    rc = os.path.join(os.path.dirname(dat), "rc")
    open(rc, "w").write(f"[run]\ndynamic_context = test_function\ndata_file = {dat}\nsource = {ROOT}\nomit = */tests/*\n")
    subprocess.run([sys.executable, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest", "tests"] + IGN,
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    cd = coverage.CoverageData(basename=dat); cd.read()
    cov = {}   # test function context -> set(file::func)
    for f in cd.measured_files():
        rel = os.path.relpath(f, ROOT)
        if not (rel.startswith("track") and rel.endswith(".py")): continue
        fr = func_ranges(f)
        for line, ctxs in cd.contexts_by_lineno(f).items():
            names = [n for (a, b, n) in fr if a < line <= b]   # a<line: the def line itself runs at import, ignore
            if not names: continue
            for c in ctxs:
                if not c: continue
                cov.setdefault(c.split("|")[0], set()).add(f"{rel}::{names[-1]}")
    return cov

if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    total_loc, files = loc()
    med, raw = timed_run(repeats)
    cov = coverage_matrix()
    # map test node id -> coverage context (context is 'module.test_name' or 'module.Class.test_name')
    tests = []
    for (cls, name), t in sorted(med.items()):
        ctx = cls.removeprefix("tests.") + "." + name.split("[")[0]   # coverage context has no "tests." prefix
        funcs = sorted(cov.get(ctx, set()))
        tests.append({"id": nodeid(cls, name), "median_s": t, "runs_s": raw[(cls, name)], "functions": funcs})
    allf = sorted(set().union(*[set(t["functions"]) for t in tests]))
    out = {"loc_source": total_loc, "loc_by_file": files, "repeats": repeats, "n_tests": len(tests),
           "n_requirements_functions": len(allf), "requirements": allf, "tests": tests,
           "suite_total_s": sum(t["median_s"] for t in tests)}
    json.dump(out, open(os.path.join(ROOT, "results", "trackJ_profile.json"), "w"), indent=1)
    print(f"LOC={total_loc} tests={len(tests)} functions covered={len(allf)} suite_s={out['suite_total_s']:.2f}")
