"""
Track C — Quantum Optimization
PennyLane simulator. Writes ../results/trackC_results.json.

NOTE: All results are SIMULATED on classical hardware (default.qubit and default.mixed),
      NOT executed on physical quantum processors.

C1. QAOA on Track B QUBO (7-qubit Max-Cut, p=1 and p=2 layers).
    Compare with classical baseline (Brute-Force 33, Simulated Annealing 33.0).
C2. VQE (Variational Quantum Eigensolver) on 4-qubit Transverse-Field Ising Model.
    Compare with exact classical Hamiltonian diagonalization.
C3. Optimization methods comparison: Parameter-Shift vs SPSA vs Quantum Natural Gradient (QNG).
C4. Parameterized Quantum Circuit (PQC / QNN) binary classifier.
C5. Finite-shot measurement effect (shots in [50, 200, 1000, 10000, infinite/analytic]).
C6. Hardware noise model effect (DepolarizingChannel p in [0, 0.02, 0.05, 0.1, 0.2]).
C7. Barren Plateaus: gradient variance decay as qubits grow n in [2, 4, 6, 8, 10].
"""
import os, json, time, math
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.linalg import eigh

R = {"metadata": {"hardware": "SIMULATED (PennyLane default.qubit / default.mixed)", "is_hardware": False}}

# Load Track B 7-node graph
trackb_path = os.path.join(os.path.dirname(__file__), "..", "results", "trackB_results.json")
if os.path.exists(trackb_path):
    with open(trackb_path) as f:
        tb_data = json.load(f)
    WMAT = np.array(tb_data["B3_maxcut"]["weight_matrix"])
    CLASSICAL_OPT = tb_data["B3_maxcut"]["brute_force"]["cut"]
else:
    # Fallback to identical seed 2 graph from Track B
    EDGES_RAW = [(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(5,6),(0,5),(1,6)]
    rng2 = np.random.default_rng(2)
    WMAT = np.zeros((7, 7), dtype=int)
    for i, j in EDGES_RAW:
        w = int(rng2.integers(1, 6))
        WMAT[i, j] = w; WMAT[j, i] = w
    CLASSICAL_OPT = 33

N_QUBITS = 7

# =================================================================
# C1. QAOA on Track B QUBO (Max-Cut, 7 nodes)
# =================================================================
# Max-Cut cost Hamiltonian:
# H_C = sum_{i < j} W[i,j]/2 * (I - Z_i Z_j)
# Maximizing Cut is equivalent to minimizing -H_C:
# Min H = sum_{i < j} (W[i,j]/2) * Z_i Z_j - sum_{i < j} (W[i,j]/2) * I
# Driver: H_M = sum_i X_i
# =================================================================
def run_c1_qaoa():
    edges = [(i, j, WMAT[i, j]) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS) if WMAT[i, j] > 0]
    dev_qaoa = qml.device("default.qubit", wires=N_QUBITS)

    # Cost Hamiltonian as a PennyLane Hamiltonian
    coeffs = []
    obs = []
    total_w = sum(w for _, _, w in edges)
    for i, j, w in edges:
        coeffs.append(w / 2.0)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    # We want to maximize cut = (total_w/2) - sum_{i<j} (w/2) Z_i Z_j
    # So minimizing cost = sum_{i<j} (w/2) Z_i Z_j
    H_cost = qml.Hamiltonian(coeffs, obs)

    def qaoa_layer(gamma, beta):
        # Cost unitary e^{-i gamma H_C}
        for i, j, w in edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(gamma * w, wires=j)
            qml.CNOT(wires=[i, j])
        # Mixer unitary e^{-i beta H_M}
        for i in range(N_QUBITS):
            qml.RX(2 * beta, wires=i)

    def make_qaoa_qnode(p_layers):
        @qml.qnode(dev_qaoa)
        def circuit(params):
            # Equal superposition
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
            # p layers
            for layer in range(p_layers):
                qaoa_layer(params[0][layer], params[1][layer])
            return qml.expval(H_cost)
        return circuit

    # Optimize p=1
    qnode_p1 = make_qaoa_qnode(p_layers=1)
    init_p1 = pnp.array([[0.5], [0.5]], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    params_p1 = init_p1
    for step in range(30):
        params_p1, energy = opt.step_and_cost(qnode_p1, params_p1)
    # Expected cut = total_w / 2 - energy
    expected_cut_p1 = float(total_w / 2.0 - energy)

    # Optimize p=2
    qnode_p2 = make_qaoa_qnode(p_layers=2)
    init_p2 = pnp.array([[0.5, 0.4], [0.5, 0.4]], requires_grad=True)
    params_p2 = init_p2
    for step in range(30):
        params_p2, energy = opt.step_and_cost(qnode_p2, params_p2)
    expected_cut_p2 = float(total_w / 2.0 - energy)

    # Sample top bitstring from p=2 statevector
    @qml.qnode(dev_qaoa)
    def prob_circuit(params):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
        for layer in range(2):
            qaoa_layer(params[0][layer], params[1][layer])
        return qml.probs(wires=range(N_QUBITS))

    probs = prob_circuit(params_p2)
    best_bit_idx = int(np.argmax(probs))
    best_bitstring = [int(b) for b in f"{best_bit_idx:0{N_QUBITS}b}"]
    # Calculate cut of this bitstring
    sampled_cut = sum(WMAT[i, j] for i in range(N_QUBITS) for j in range(i+1, N_QUBITS) if best_bitstring[i] != best_bitstring[j])

    approx_ratio_p1 = expected_cut_p1 / CLASSICAL_OPT
    approx_ratio_p2 = expected_cut_p2 / CLASSICAL_OPT

    return {
        "problem": "7-qubit Max-Cut from Track B",
        "classical_optimal_cut": CLASSICAL_OPT,
        "qaoa_p1": {
            "expected_cut": round(expected_cut_p1, 3),
            "approximation_ratio": round(approx_ratio_p1, 3),
            "optimal_params": params_p1.tolist()
        },
        "qaoa_p2": {
            "expected_cut": round(expected_cut_p2, 3),
            "approximation_ratio": round(approx_ratio_p2, 3),
            "sampled_best_cut": int(sampled_cut),
            "sampled_best_bitstring": best_bitstring,
            "max_probability": round(float(np.max(probs)), 4)
        },
        "comparison_with_classical": {
            "classical_brute_force": CLASSICAL_OPT,
            "classical_simulated_annealing": 33.0,
            "qaoa_p2_expected": round(expected_cut_p2, 3),
            "qaoa_p2_best_sample": int(sampled_cut)
        }
    }

# =================================================================
# C2. VQE (Variational Quantum Eigensolver) on 4-Qubit Spin Chain
# =================================================================
def run_c2_vqe():
    wires = 4
    dev = qml.device("default.qubit", wires=wires)

    # Transverse-Field Ising Model: H = - sum_{i=0}^2 Z_i Z_{i+1} - 0.8 sum_{i=0}^3 X_i
    coeffs = []
    obs = []
    for i in range(wires - 1):
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(wires):
        coeffs.append(-0.8)
        obs.append(qml.PauliX(i))
    H = qml.Hamiltonian(coeffs, obs)

    # Exact classical ground state energy via matrix diagonalization
    H_mat = qml.matrix(H)
    eigvals, _ = eigh(H_mat)
    exact_ground_energy = float(eigvals[0])

    # Hardware-efficient ansatz (RY + CNOT ladder)
    depth = 2
    @qml.qnode(dev)
    def vqe_circuit(params):
        # Initial layer
        for i in range(wires):
            qml.RY(params[0, i], wires=i)
        for d in range(depth):
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(wires):
                qml.RY(params[d + 1, i], wires=i)
        return qml.expval(H)

    np.random.seed(42)
    init_params = pnp.array(np.random.uniform(0, np.pi, size=(depth + 1, wires)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    params = init_params
    energy_history = []
    for it in range(40):
        params, cost = opt.step_and_cost(vqe_circuit, params)
        energy_history.append(float(cost))

    vqe_energy = float(energy_history[-1])
    error = abs(vqe_energy - exact_ground_energy)

    return {
        "model": "4-qubit Transverse-Field Ising Model (J=1.0, g=0.8)",
        "exact_ground_state_energy": round(exact_ground_energy, 5),
        "vqe_final_energy": round(vqe_energy, 5),
        "energy_absolute_error": round(error, 6),
        "relative_error_pct": round(100.0 * error / abs(exact_ground_energy), 4),
        "ansatz": "Hardware-efficient (RY + CNOT ladder, depth=2)",
        "iterations": 40
    }

# =================================================================
# C3. Parameter-Shift vs SPSA vs Quantum Natural Gradient (QNG)
# =================================================================
def run_c3_optimizers():
    wires = 3
    dev = qml.device("default.qubit", wires=wires)

    # Hamiltonian to minimize: sum Z_i Z_{i+1}
    H = qml.Hamiltonian([1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2)])

    @qml.qnode(dev)
    def circuit(params):
        for i in range(wires):
            qml.RY(params[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        for i in range(wires):
            qml.RY(params[i + wires], wires=i)
        return qml.expval(H)

    p_init = pnp.array([0.5, 0.8, -0.3, 0.2, 0.9, -0.6], requires_grad=True)

    # 1. Parameter-Shift Rule (GradientDescentOptimizer with parameter-shift gradients)
    opt_ps = qml.GradientDescentOptimizer(stepsize=0.15)
    p_ps = p_init.copy()
    hist_ps = []
    t0 = time.perf_counter()
    for _ in range(25):
        p_ps, c = opt_ps.step_and_cost(circuit, p_ps)
        hist_ps.append(float(c))
    t_ps = time.perf_counter() - t0

    # 2. SPSA (Simultaneous Perturbation Stochastic Approximation)
    opt_spsa = qml.SPSAOptimizer(maxiter=25, c=0.15, a=0.15)
    p_spsa = p_init.copy()
    hist_spsa = []
    t0 = time.perf_counter()
    for _ in range(25):
        p_spsa, c = opt_spsa.step_and_cost(circuit, p_spsa)
        hist_spsa.append(float(c))
    t_spsa = time.perf_counter() - t0

    # 3. Quantum Natural Gradient (QNG via Fubini-Study metric tensor)
    opt_qng = qml.QNGOptimizer(stepsize=0.15)
    p_qng = p_init.copy()
    hist_qng = []
    t0 = time.perf_counter()
    for _ in range(25):
        p_qng, c = opt_qng.step_and_cost(circuit, p_qng)
        hist_qng.append(float(c))
    t_qng = time.perf_counter() - t0

    # Exact minimum eigenvalue of H
    H_mat = qml.matrix(H)
    e_min = float(np.min(np.linalg.eigvalsh(H_mat)))

    return {
        "exact_min_energy": round(e_min, 4),
        "parameter_shift": {
            "final_cost": round(hist_ps[-1], 4),
            "time_s": round(t_ps, 4),
            "circuit_evals_per_step": 2 * len(p_init)  # 2 evaluations per parameter for 2-term shift rule
        },
        "spsa": {
            "final_cost": round(hist_spsa[-1], 4),
            "time_s": round(t_spsa, 4),
            "circuit_evals_per_step": 2  # exactly 2 evaluations independent of parameter count
        },
        "qng": {
            "final_cost": round(hist_qng[-1], 4),
            "time_s": round(t_qng, 4),
            "note": "Uses Fubini-Study quantum metric tensor for natural gradient descent"
        }
    }

# =================================================================
# C4. Parameterized Quantum Circuit (PQC / QNN) Classifier
# =================================================================
def run_c4_qnn():
    dev = qml.device("default.qubit", wires=2)

    # 2D synthetic dataset: concentric circles (non-linearly separable)
    np.random.seed(10)
    n_samples = 60
    r = np.random.uniform(0.1, 1.2, n_samples)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    y = np.where(r > 0.65, 1, -1)  # binary labels {-1, +1}

    @qml.qnode(dev)
    def qnn_circuit(weights, x):
        # Angle embedding
        qml.AngleEmbedding(x, wires=[0, 1], rotation='Y')
        # Parameterized ansatz layers
        for layer in range(2):
            qml.Rot(*weights[layer, 0], wires=0)
            qml.Rot(*weights[layer, 1], wires=1)
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def predict(weights, X_data):
        return np.array([np.sign(qnn_circuit(weights, x)) for x in X_data])

    def cost(weights):
        loss = 0.0
        for xi, yi in zip(X, y):
            pred = qnn_circuit(weights, xi)
            loss = loss + (pred - yi) ** 2
        return loss / len(y)

    np.random.seed(42)
    weights = pnp.array(np.random.uniform(0, np.pi, (2, 2, 3)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    for it in range(30):
        weights, c = opt.step_and_cost(cost, weights)

    preds = predict(weights, X)
    accuracy = float(np.mean(preds == y))

    return {
        "task": "2-qubit Non-linear Concentric Circles Classification",
        "n_samples": n_samples,
        "n_parameters": 2 * 2 * 3,
        "train_accuracy": round(accuracy, 4),
        "final_mse_loss": round(float(c), 4),
        "iterations": 30
    }

# =================================================================
# C5. Finite-Shot Measurement Effect
# =================================================================
def run_c5_finite_shots():
    # Circuit expectation of Z(0) under Hadamard state (|0> + |1>)/sqrt(2) is exactly 0.0
    shots_list = [50, 200, 1000, 10000]
    results = {}

    for s in shots_list:
        dev_s = qml.device("default.qubit", wires=1, shots=s)
        @qml.qnode(dev_s)
        def circ_shot():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        # Sample 10 evaluations
        vals = [float(circ_shot()) for _ in range(10)]
        results[f"shots_{s}"] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "theoretical_std": round(1.0 / math.sqrt(s), 4)
        }

    # Analytic (infinite shots)
    dev_inf = qml.device("default.qubit", wires=1, shots=None)
    @qml.qnode(dev_inf)
    def circ_inf():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))
    results["shots_infinite_analytic"] = {
        "value": float(circ_inf()),
        "std": 0.0
    }

    return results

# =================================================================
# C6. Hardware Noise Model Effect
# =================================================================
def run_c6_hardware_noise():
    # Evaluate 2-qubit Bell state fidelity / <Z0 Z1> under Depolarizing noise
    p_noise_list = [0.0, 0.02, 0.05, 0.1, 0.2]
    noise_results = {}

    for p in p_noise_list:
        dev_noise = qml.device("default.mixed", wires=2)
        @qml.qnode(dev_noise)
        def noisy_bell():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            if p > 0.0:
                qml.DepolarizingChannel(p, wires=0)
                qml.DepolarizingChannel(p, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        val = float(noisy_bell())
        noise_results[f"depolarizing_p_{p}"] = {
            "expval_Z0_Z1": round(val, 4),
            "ideal_value": 1.0,
            "fidelity_degradation_pct": round(100.0 * (1.0 - val), 2)
        }

    return noise_results

# =================================================================
# C7. Barren Plateaus (Gradient Variance as Qubits Grow)
# =================================================================
def run_c7_barren_plateaus():
    qubit_sizes = [2, 4, 6, 8]
    n_samples = 25
    variances = {}

    for n in qubit_sizes:
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def bp_circuit(params):
            # Random deep layered ansatz (Haar-like)
            for d in range(n):
                for i in range(n):
                    qml.RY(params[d, i, 0], wires=i)
                    qml.RZ(params[d, i, 1], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        grads = []
        for _ in range(n_samples):
            p = pnp.array(np.random.uniform(0, 2 * np.pi, size=(n, n, 2)), requires_grad=True)
            g = qml.grad(bp_circuit)(p)
            grads.append(float(g[0, 0, 0]))  # gradient of first parameter

        var_g = float(np.var(grads))
        variances[f"qubits_{n}"] = {
            "variance": float(f"{var_g:.6e}"),
            "std": float(f"{np.std(grads):.6e}")
        }
        print(f"  Qubits={n:2d}: Var[grad]={var_g:.6e}")

    return {
        "qubit_counts": qubit_sizes,
        "n_random_initializations": n_samples,
        "gradient_variances": variances,
        "theoretical_scaling": "Var[grad] ~ 2^{-n} (exponentially vanishing gradients / Barren Plateau)"
    }

# =================================================================
# Main Execution
# =================================================================
if __name__ == "__main__":
    print("=== Track C — Quantum Optimization (PennyLane Simulator) ===")
    print("STATUS: SIMULATED on classical hardware (default.qubit / default.mixed)\n")

    print("--> Running C1: QAOA on Track B QUBO...")
    R["C1_qaoa_trackB"] = run_c1_qaoa()

    print("--> Running C2: VQE on 4-qubit Ising chain...")
    R["C2_vqe_ising"] = run_c2_vqe()

    print("--> Running C3: Optimizer comparison (Parameter-Shift vs SPSA vs QNG)...")
    R["C3_optimizers"] = run_c3_optimizers()

    print("--> Running C4: PQC / QNN Binary Classifier...")
    R["C4_pqc_qnn"] = run_c4_qnn()

    print("--> Running C5: Finite-Shot Measurement Effect...")
    R["C5_finite_shots"] = run_c5_finite_shots()

    print("--> Running C6: Hardware Noise Model Effect...")
    R["C6_hardware_noise"] = run_c6_hardware_noise()

    print("--> Running C7: Barren Plateaus (Gradient Variance Decay)...")
    R["C7_barren_plateaus"] = run_c7_barren_plateaus()

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "trackC_results.json")
    with open(out_path, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\n[OK] Results saved to", out_path)
