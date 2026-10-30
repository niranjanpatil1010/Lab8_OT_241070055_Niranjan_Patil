"""
Track F — Compiler Optimization Benchmarks and Register Allocation Experiment.
Writes ../results/trackF_results.json.
"""
import json, os, time
import numpy as np
from compiler import (
    tokenize, Parser, IRGenerator, VirtualMachine,
    optimize_all, analyze_live_ranges, build_interference_graph,
    allocate_registers_greedy, allocate_registers_ilp
)

R = {}

# Program 1: Constant expressions, common subexpressions, dead variables
PROG_1 = """
a = 10 + 5;
b = a * 2;
c = 10 + 5;
dead1 = 99 * 4;
d = b + c;
print(d);
"""

# Program 2: Loop with invariant code, strength reduction, dead variables
PROG_2 = """
x = 5;
inv = 100 * 2;
counter = 4;
sum = 0;
while (counter) {
    unused = 50 + 20;
    step = 2 * 1;
    sum = sum + step;
    counter = counter - 1;
}
print(sum);
"""

# Program 3: Multi-variable register pressure benchmark
PROG_3 = """
v1 = 1;
v2 = 2;
v3 = 3;
v4 = 4;
v5 = 5;
v6 = 6;
v7 = v1 + v2;
v8 = v3 + v4;
v9 = v5 + v6;
ans = v7 + v8;
ans = ans + v9;
print(ans);
"""

PROGRAMS = [
    ("Program1_Expressions", PROG_1),
    ("Program2_Loop_Invariants", PROG_2),
    ("Program3_Register_Pressure", PROG_3)
]

def benchmark_program(name, source_code):
    # 1. Compile to Unoptimized IR
    tokens = tokenize(source_code)
    parser = Parser(tokens)
    ast = parser.parse()
    ir_gen = IRGenerator()
    unopt_ir = ir_gen.generate(ast)

    # 2. Run Unoptimized
    vm_unopt = VirtualMachine()
    t0 = time.perf_counter()
    out_unopt, _ = vm_unopt.run(unopt_ir)
    t_run_unopt = time.perf_counter() - t0
    cycles_unopt = vm_unopt.instruction_exec_count
    code_size_unopt = len(unopt_ir)

    # 3. Optimize IR
    opt_ir, compile_time = optimize_all(unopt_ir)

    # 4. Run Optimized
    vm_opt = VirtualMachine()
    t0 = time.perf_counter()
    out_opt, _ = vm_opt.run(opt_ir)
    t_run_opt = time.perf_counter() - t0
    cycles_opt = vm_opt.instruction_exec_count
    code_size_opt = len(opt_ir)

    # Correctness check: Outputs must be identical!
    assert out_unopt == out_opt, f"Correctness failure in {name}: {out_unopt} != {out_opt}"

    # 5. Register Allocation Experiment on this program's variables
    intervals = analyze_live_ranges(opt_ir)
    vars_list, adj = build_interference_graph(intervals)
    k_regs = 3  # pressure test with 3 physical registers

    color_greedy, spills_greedy = allocate_registers_greedy(vars_list, adj, k_registers=k_regs)
    color_ilp, spills_ilp = allocate_registers_ilp(vars_list, adj, k_registers=k_regs)

    # ILP is optimal: spills_ilp <= spills_greedy
    assert spills_ilp <= spills_greedy, f"ILP spills ({spills_ilp}) > Greedy spills ({spills_greedy})"

    return {
        "name": name,
        "correctness": "PASS (Outputs identical: %s)" % out_opt,
        "output": out_opt,
        "unoptimized": {
            "code_size_instructions": code_size_unopt,
            "dynamic_cycles_executed": cycles_unopt,
            "runtime_s": round(t_run_unopt, 6)
        },
        "optimized": {
            "code_size_instructions": code_size_opt,
            "dynamic_cycles_executed": cycles_opt,
            "runtime_s": round(t_run_opt, 6),
            "compile_time_s": round(compile_time, 6)
        },
        "improvement": {
            "code_size_reduction_pct": round(100.0 * (code_size_unopt - code_size_opt) / code_size_unopt, 2),
            "cycle_reduction_pct": round(100.0 * (cycles_unopt - cycles_opt) / cycles_unopt, 2),
        },
        "register_allocation": {
            "num_variables": len(vars_list),
            "physical_registers_k": k_regs,
            "spills_greedy": spills_greedy,
            "spills_ilp": spills_ilp,
            "ilp_improvement_over_greedy": spills_greedy - spills_ilp
        }
    }

def run_register_allocation_experiment():
    """
    Dedicated Register Allocation comparison on 8-variable interference graphs:
    Greedy (Chaitin / Welsh-Powell heuristic) vs ILP (scipy.optimize.milp).
    Evaluated across 10 random interference graph topologies (seed 0..9).
    """
    results = {}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 8
        adj = (rng.random((n, n)) < 0.45).astype(int)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        vars_l = [f"r{i}" for i in range(n)]

        for k in [2, 3]:
            cg, sg = allocate_registers_greedy(vars_l, adj, k_registers=k)
            ci, si = allocate_registers_ilp(vars_l, adj, k_registers=k)
            key = f"graph_seed_{seed}_k_{k}"
            results[key] = {
                "variables": n,
                "edges": int(np.sum(adj) // 2),
                "k_registers": k,
                "greedy_spills": sg,
                "ilp_spills": si,
                "spill_reduction_ilp": sg - si
            }
    return results

if __name__ == "__main__":
    print("=== Track F — Compiler Optimization ===")
    results = {}
    for name, code in PROGRAMS:
        print(f"--> Benchmarking {name}...")
        res = benchmark_program(name, code)
        results[name] = res
        print(f"    Code size: {res['unoptimized']['code_size_instructions']} -> {res['optimized']['code_size_instructions']} instrs (-{res['improvement']['code_size_reduction_pct']}%)")
        print(f"    Dynamic cycles: {res['unoptimized']['dynamic_cycles_executed']} -> {res['optimized']['dynamic_cycles_executed']} (-{res['improvement']['cycle_reduction_pct']}%)")
        print(f"    Verification: {res['correctness']}\n")

    print("--> Running Register Allocation Optimization Experiment (Greedy vs ILP)...")
    reg_alloc_results = run_register_allocation_experiment()
    R["benchmarks"] = results
    R["register_allocation_experiment"] = reg_alloc_results

    # Summary of register allocation
    tot_greedy_spills = sum(v["greedy_spills"] for v in reg_alloc_results.values())
    tot_ilp_spills = sum(v["ilp_spills"] for v in reg_alloc_results.values())
    print(f"    Total spills across 20 configurations: Greedy = {tot_greedy_spills}, ILP = {tot_ilp_spills}")
    print(f"    ILP reduced spills by {tot_greedy_spills - tot_ilp_spills} ({round(100*(tot_greedy_spills-tot_ilp_spills)/tot_greedy_spills, 2)}% reduction)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "trackF_results.json")
    with open(out_path, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\n[OK] Results saved to", out_path)
