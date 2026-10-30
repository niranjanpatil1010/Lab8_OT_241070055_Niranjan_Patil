"""Tests for Track H — DBMS Optimization.
Run from repo root: python -m pytest tests/test_trackH.py -v
"""
import pytest, os, sqlite3
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackH'))
from run import (
    generate_dataset, execute_workload, get_explain_plan,
    index_selection_greedy, index_selection_dp, index_selection_ilp,
    QUERIES, CANDIDATE_INDEXES, DB_PATH
)

class TestTrackHDBMS:
    @pytest.fixture(scope="class")
    def setup_db(self):
        # Create small test db
        test_db = os.path.join(os.path.dirname(__file__), "test_h.db")
        generate_dataset(test_db, seed=123)
        yield test_db
        if os.path.exists(test_db):
            os.remove(test_db)

    def test_database_creation_and_integrity(self, setup_db):
        conn = sqlite3.connect(setup_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM customers;")
        assert cur.fetchone()[0] == 10000
        cur.execute("SELECT COUNT(*) FROM orders;")
        assert cur.fetchone()[0] == 30000
        cur.execute("SELECT COUNT(*) FROM lineitems;")
        assert cur.fetchone()[0] == 60000
        conn.close()

    def test_query_workload_execution(self, setup_db):
        conn = sqlite3.connect(setup_db)
        mean_lat, std_lat, results = execute_workload(conn, repetitions=2)
        assert mean_lat > 0.0
        assert len(results) == len(QUERIES)
        conn.close()

    def test_explain_plan_generation(self, setup_db):
        conn = sqlite3.connect(setup_db)
        plan = get_explain_plan(conn, QUERIES[0][1])
        assert "SCAN" in plan or "SEARCH" in plan
        conn.close()

    def test_index_selection_knapsack_algorithms_agree(self):
        # Synthetic metrics in KB units
        metrics = [
            {"name": "idx1", "cost_bytes": 1024, "benefit_ms": 10.0, "ratio": 10.0},
            {"name": "idx2", "cost_bytes": 2048, "benefit_ms": 15.0, "ratio": 7.5},
            {"name": "idx3", "cost_bytes": 1024, "benefit_ms": 12.0, "ratio": 8.0},
            {"name": "idx4", "cost_bytes": 3072, "benefit_ms": 25.0, "ratio": 8.33},
        ]
        budget = 4096
        sel_dp, ben_dp, cost_dp = index_selection_dp(metrics, budget)
        sel_ilp, ben_ilp, cost_ilp = index_selection_ilp(metrics, budget)

        assert cost_dp <= budget
        assert cost_ilp <= budget
        # Both exact methods should find optimal benefit
        assert abs(ben_dp - ben_ilp) < 1.0

    def test_query_results_strictly_identical_after_index(self, setup_db):
        conn = sqlite3.connect(setup_db)
        # Run baseline
        _, _, base_res = execute_workload(conn, repetitions=1)
        # Create an index
        conn.execute("CREATE INDEX test_idx_cust ON orders(customer_id);")
        conn.commit()
        # Run indexed
        _, _, opt_res = execute_workload(conn, repetitions=1)
        # Results must be identical
        for q in base_res:
            assert base_res[q] == opt_res[q]
        conn.close()
