"""Every track has its own trackX/run.py, and the tests do `from run import ...`.
Python caches one module called 'run', so running the whole suite in ONE pytest session (or a subset of node ids
from several tracks) imported the wrong file (FAILURE_LOG #17). Fix: tests/test_trackX.py is always bound to
trackX/run.py, loaded by path, both when the test module is imported (collection) and while its tests execute
(some tests import lazily inside the test body)."""
import importlib.util, os, sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_cache = {}

def _run_module_for(stem):
    track = stem.replace("test_", "", 1)               # test_trackD -> trackD
    path = os.path.join(ROOT, track, "run.py")
    if not os.path.exists(path):
        return None
    if path not in _cache:
        d = os.path.dirname(path)
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("run", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run"] = mod
        spec.loader.exec_module(mod)
        _cache[path] = mod
    return _cache[path]

def _imports_run(stem):
    f = os.path.join(ROOT, "tests", stem + ".py")
    return os.path.exists(f) and ("from run import" in open(f).read() or "import run" in open(f).read())

def _bind(stem):
    if not _imports_run(stem):      # e.g. test_trackA validates saved results; trackA/run.py is a script with side effects
        return
    try:
        mod = _run_module_for(stem)
    except Exception:
        mod = None
    if mod is not None:
        sys.modules["run"] = mod
    else:
        sys.modules.pop("run", None)

def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _bind(collector.path.stem)

@pytest.fixture(autouse=True)
def _own_run_module(request):
    _bind(os.path.splitext(os.path.basename(str(request.node.fspath)))[0])
    yield
