"""Validates Track D5/D6 (data locality, synchronization/communication): saved results + a fresh correctness check."""
import importlib.util, json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = json.load(open(os.path.join(ROOT, "results", "trackD_locality_sync.json")))

def _mod():
    spec = importlib.util.spec_from_file_location("run_locality_sync", os.path.join(ROOT, "trackD", "run_locality_sync.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_column_traversal_slower_than_row():
    d = R["D5_data_locality"]
    assert d["sum_cols_s"] > d["sum_rows_s"] and d["col_over_row_slowdown"] > 1.0

def test_all_timings_positive():
    d = R["D5_data_locality"]["transpose_by_tile_s"]
    assert all(v > 0 for v in d.values()) and len(d) == 8

def test_lock_per_element_slower_than_local_partial_sums():
    d = R["D6_sync_comm"]
    assert d["lock_per_element_s"] > d["local_partial_sums_s"]

def test_tiny_chunks_cost_more_than_large_chunks():
    g = R["D6_sync_comm"]["pool_map_chunksize_s"]
    assert g["1"] > g["1000"]

def test_helpers_are_correct():
    m = _mod()
    assert m._work_local((0, 1000)) == sum(range(1000)) and m._tiny(21) == 42
