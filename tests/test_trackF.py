"""Tests for Track F — Compiler Optimization.
Run from repo root: python -m pytest tests/test_trackF.py -v
"""
import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trackF'))
from compiler import (
    tokenize, Parser, IRGenerator, VirtualMachine,
    pass_constant_folding, pass_strength_reduction, pass_cse, pass_dce, pass_licm,
    optimize_all, analyze_live_ranges, build_interference_graph,
    allocate_registers_greedy, allocate_registers_ilp
)

class TestTrackFCompiler:
    def test_lexer_and_parser(self):
        code = "a = 5 + 3 * 2; print(a);"
        tokens = tokenize(code)
        assert len(tokens) > 5
        parser = Parser(tokens)
        ast = parser.parse()
        assert len(ast) == 2

    def test_unoptimized_execution(self):
        code = "x = 10; y = 20; z = x + y; print(z);"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        ir = IRGenerator().generate(ast)
        vm = VirtualMachine()
        out, _ = vm.run(ir)
        assert out == [30]

    def test_constant_folding(self):
        code = "x = 4 + 6; print(x);"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        ir = IRGenerator().generate(ast)
        ir_opt = pass_constant_folding(ir)
        # Should have folded 4 + 6 -> 10
        vm = VirtualMachine()
        out, _ = vm.run(ir_opt)
        assert out == [10]

    def test_strength_reduction(self):
        code = "a = 7; b = a * 2; c = a * 0; d = a * 1; print(b); print(c); print(d);"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        ir = IRGenerator().generate(ast)
        ir_sr = pass_strength_reduction(ir)
        vm = VirtualMachine()
        out, _ = vm.run(ir_sr)
        assert out == [14, 0, 7]

    def test_cse_elimination(self):
        code = "a = 5; b = 3; c = a + b; d = a + b; print(c); print(d);"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        ir = IRGenerator().generate(ast)
        ir_cse = pass_cse(ir)
        # Second (a + b) must be replaced by assignment
        vm = VirtualMachine()
        out, _ = vm.run(ir_cse)
        assert out == [8, 8]

    def test_dce_elimination(self):
        code = "used = 10; dead_var = 999; print(used);"
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        ir = IRGenerator().generate(ast)
        ir_dce = pass_dce(ir)
        # dead_var should not be computed
        res_vars = [instr.res for instr in ir_dce if instr.res is not None]
        assert 'dead_var' not in res_vars

    def test_optimizer_preserves_program_semantics(self):
        # Program with loop, invariants, dead code
        code = """
        acc = 0;
        k = 3;
        while (k) {
            step = 5;
            acc = acc + step;
            k = k - 1;
        }
        print(acc);
        """
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        unopt_ir = IRGenerator().generate(ast)
        opt_ir, _ = optimize_all(unopt_ir)

        vm1 = VirtualMachine()
        out1, _ = vm1.run(unopt_ir)
        vm2 = VirtualMachine()
        out2, _ = vm2.run(opt_ir)

        assert out1 == [15]
        assert out2 == [15]
        assert vm2.instruction_exec_count < vm1.instruction_exec_count

    def test_register_allocation_ilp_validity_and_optimality(self):
        # 6 variables with conflict triangle
        vars_l = ['a', 'b', 'c', 'd', 'e']
        adj = np.zeros((5, 5), dtype=int)
        # Create complete graph K5
        for i in range(5):
            for j in range(i + 1, 5):
                adj[i, j] = 1; adj[j, i] = 1

        k_regs = 3
        # For K5 with 3 registers, at least 5 - 3 = 2 variables must be spilled
        cmap, spills_ilp = allocate_registers_ilp(vars_l, adj, k_registers=k_regs)
        assert spills_ilp == 2

        # Check validity: no two adjacent variables have the same non-negative register
        for i in range(5):
            for j in range(i + 1, 5):
                if adj[i, j] == 1:
                    r_i = cmap[vars_l[i]]
                    r_j = cmap[vars_l[j]]
                    if r_i != -1 and r_j != -1:
                        assert r_i != r_j, f"Interference conflict between {vars_l[i]} and {vars_l[j]}"
