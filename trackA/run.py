"""Track A - continuous optimization. numpy only, seeded. Writes ../results/trackA_results.json"""
import json, numpy as np
rng0 = np.random.default_rng(0)
R = {}

# ---------- 1. deterministic optimizers on convex / non-convex objectives ----------
def make_quad():
    A = np.diag([1.0, 20.0])                     # convex, condition number 20
    return (lambda x: 0.5 * x @ A @ x, lambda x: A @ x, lambda x: A, np.array([5.0, 5.0]))
def make_rosen():                                # non-convex valley, min at (1,1)
    f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = lambda x: np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]), 200*(x[1]-x[0]**2)])
    H = lambda x: np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200.0]])
    return f, g, H, np.array([-1.2, 1.0])

def run_opt(name, f, g, H, x0, lr, iters=2000, tol=1e-8):
    x = x0.copy(); v = np.zeros_like(x); s = np.zeros_like(x); m = np.zeros_like(x); b1, b2, eps = .9, .999, 1e-8
    for k in range(1, iters + 1):
        if name == "GD": x = x - lr * g(x)
        elif name == "Momentum": v = .9*v + g(x); x = x - lr * v
        elif name == "NAG": v = .9*v + g(x - lr*.9*v); x = x - lr * v
        elif name == "AdaGrad": gr = g(x); s += gr**2; x = x - lr * gr / (np.sqrt(s) + eps)
        elif name == "RMSProp": gr = g(x); s = .9*s + .1*gr**2; x = x - lr * gr / (np.sqrt(s) + eps)
        elif name == "Adam":
            gr = g(x); m = b1*m + (1-b1)*gr; s = b2*s + (1-b2)*gr**2
            x = x - lr * (m/(1-b1**k)) / (np.sqrt(s/(1-b2**k)) + eps)
        elif name == "Newton": x = x - np.linalg.solve(H(x) + 1e-8*np.eye(len(x)), g(x))
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e8: return None, k
        if f(x) < tol: return float(f(x)), k
    return float(f(x)), iters

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
det = {}
for pname, mk in [("convex_quadratic", make_quad), ("rosenbrock_nonconvex", make_rosen)]:
    f, g, H, x0 = mk(); det[pname] = {}
    for name in ["GD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "Newton"]:
        grid = {}
        for lr in ([1.0] if name == "Newton" else LRS):
            fv, it = run_opt(name, f, g, H, x0, lr)
            grid[lr] = dict(final_f=fv, iters=it)
        ok = {lr: r for lr, r in grid.items() if r["final_f"] is not None}
        best = min(ok, key=lambda lr: (ok[lr]["final_f"] if ok[lr]["iters"] == 2000 else 0, ok[lr]["iters"])) if ok else None
        det[pname][name] = dict(best_lr=best, **(ok[best] if best else {}),
                                n_lr_diverged=sum(r["final_f"] is None for r in grid.values()),
                                n_lr_converged=sum(r["final_f"] is not None and r["iters"] < 2000 for r in grid.values()))
R["deterministic"] = det

# ---------- 2. noisy objective: GD vs SGD vs mini-batch SGD on linear regression ----------
n, d = 2000, 5
X = rng0.normal(size=(n, d)); w_true = rng0.normal(size=d); y = X @ w_true + rng0.normal(scale=1.0, size=n)
w_opt = np.linalg.lstsq(X, y, rcond=None)[0]; L = lambda w: float(np.mean((X @ w - y)**2) / 2)
def sgd(bs, lr, epochs=20, seed=0, decay=False):
    r = np.random.default_rng(seed); w = np.zeros(d); t = 0
    for ep in range(epochs):
        idx = r.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i+bs]; t += 1; step = lr / (1 + 0.01*t) if decay else lr
            w -= step * X[b].T @ (X[b] @ w - y[b]) / len(b)
    return L(w) - L(w_opt)
noisy = {}
for label, bs, lr in [("GD(full batch)", n, 0.1), ("SGD(bs=1)", 1, 0.01), ("minibatch(bs=32)", 32, 0.05)]:
    runs = [sgd(bs, lr, seed=s) for s in range(10)]; noisy[label] = dict(mean_gap=float(np.mean(runs)), std_gap=float(np.std(runs)))
noisy["SGD(bs=1) + lr decay"] = dict(mean_gap=float(np.mean([sgd(1, .05, seed=s, decay=True) for s in range(10)])),
                                     std_gap=float(np.std([sgd(1, .05, seed=s, decay=True) for s in range(10)])))
R["noisy_objective_gap_to_optimum_loss"] = noisy

# ---------- 3. Gauss-Newton vs GD on nonlinear least squares y = a*exp(b*x) ----------
xs = np.linspace(0, 2, 40); ys = 2.0*np.exp(0.7*xs) + rng0.normal(scale=0.05, size=40)
res = lambda p: p[0]*np.exp(p[1]*xs) - ys
J = lambda p: np.stack([np.exp(p[1]*xs), p[0]*xs*np.exp(p[1]*xs)], 1)
cost = lambda p: float(0.5*np.sum(res(p)**2))
p = np.array([1.0, 0.1]); gn = []
for k in range(10): p = p - np.linalg.solve(J(p).T @ J(p) + 1e-9*np.eye(2), J(p).T @ res(p)); gn.append(cost(p))
p2 = np.array([1.0, 0.1]); gd = []
for k in range(10): p2 = p2 - 1e-3 * J(p2).T @ res(p2); gd.append(cost(p2))
R["gauss_newton"] = dict(GN_cost_by_iter=gn, GD_cost_by_iter=gd, GN_params=p.tolist())

# ---------- 4. backprop on a 2-layer MLP + numerical gradient check ----------
m_ = 200; Xc = rng0.normal(size=(m_, 2)); yc = ((Xc[:, 0]*Xc[:, 1]) > 0).astype(float)   # XOR-like, not linearly separable
def init(h=8, seed=1):
    r = np.random.default_rng(seed); return dict(W1=r.normal(scale=.7, size=(2, h)), b1=np.zeros(h), W2=r.normal(scale=.7, size=(h, 1)), b2=np.zeros(1))
def fwd(P, X):
    z1 = X @ P["W1"] + P["b1"]; a1 = np.tanh(z1); z2 = (a1 @ P["W2"] + P["b2"]).ravel(); return z1, a1, 1/(1+np.exp(-z2))
def loss_grads(P, X, y, lam=1e-3):
    z1, a1, p = fwd(P, X); N = len(y); eps = 1e-12
    loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)) + lam*(np.sum(P["W1"]**2) + np.sum(P["W2"]**2))   # + L2 regularisation
    dz2 = (p - y)[:, None] / N                      # backprop: output layer
    G = dict(W2=a1.T @ dz2 + 2*lam*P["W2"], b2=dz2.sum(0))
    dz1 = (dz2 @ P["W2"].T) * (1 - a1**2)           # backprop: through tanh
    G["W1"] = X.T @ dz1 + 2*lam*P["W1"]; G["b1"] = dz1.sum(0); return loss, G
P = init(); _, G = loss_grads(P, Xc, yc); maxerr = 0.0
for k in P:
    it = np.nditer(P[k], flags=["multi_index"])
    for _ in it:
        i = it.multi_index; old = P[k][i]; h = 1e-5
        P[k][i] = old + h; lp, _ = loss_grads(P, Xc, yc); P[k][i] = old - h; lm, _ = loss_grads(P, Xc, yc); P[k][i] = old
        num = (lp - lm)/(2*h); maxerr = max(maxerr, abs(num - G[k][i]) / max(1e-8, abs(num) + abs(G[k][i])))
mlp = {}
for oname, lr in [("GD", 0.5), ("Adam", 0.05)]:
    P = init(); m = {k: np.zeros_like(v) for k, v in P.items()}; s = {k: np.zeros_like(v) for k, v in P.items()}
    for t in range(1, 1501):
        l, G = loss_grads(P, Xc, yc)
        for k in P:
            if oname == "GD": P[k] -= lr * G[k]
            else:
                m[k] = .9*m[k] + .1*G[k]; s[k] = .999*s[k] + .001*G[k]**2
                P[k] -= lr * (m[k]/(1-.9**t)) / (np.sqrt(s[k]/(1-.999**t)) + 1e-8)
    mlp[oname] = dict(final_loss=float(l), accuracy=float(np.mean((fwd(P, Xc)[2] > .5) == yc)))
R["backprop_mlp"] = dict(max_relative_gradient_check_error=float(maxerr), results=mlp)

# ---------- 5. curvature: Hessian-vector products, Fisher diag vs Hessian diag, low-rank via power iteration ----------
Xl = rng0.normal(size=(500, 6)); wt = rng0.normal(size=6); yl = (rng0.random(500) < 1/(1+np.exp(-Xl@wt))).astype(float)
def lgrad(w): pr = 1/(1+np.exp(-Xl@w)); return Xl.T @ (pr - yl) / len(yl)
def lhess(w): pr = 1/(1+np.exp(-Xl@w)); return (Xl.T * (pr*(1-pr))) @ Xl / len(yl)
w0 = rng0.normal(size=6)*.3
hvp = lambda v, h=1e-5: (lgrad(w0 + h*v) - lgrad(w0 - h*v)) / (2*h)          # Hv without forming H
Hx = lhess(w0); v = rng0.normal(size=6)
v_it = rng0.normal(size=6)
for _ in range(200): v_it = hvp(v_it); v_it /= np.linalg.norm(v_it)             # power iteration -> top eigenvalue (rank-1 curvature)
pr = 1/(1+np.exp(-Xl@w0)); fisher_diag = np.mean((Xl*(pr-yl)[:, None])**2, 0)  # empirical Fisher diagonal
R["curvature"] = dict(hvp_vs_exact_max_abs_err=float(np.abs(hvp(v) - Hx@v).max()),
                      top_eig_power_iter_via_HVP=float(v_it @ hvp(v_it)), top_eig_exact=float(np.linalg.eigvalsh(Hx)[-1]),
                      hessian_diag=np.diag(Hx).round(4).tolist(), empirical_fisher_diag=fisher_diag.round(4).tolist())

# ---------- 6. landscapes: saddle, local minima ----------
sad_f = lambda x: x[0]**2 - x[1]**2 + .25*x[1]**4; sad_g = lambda x: np.array([2*x[0], -2*x[1] + x[1]**3])
def gd_path(g, x, lr=.05, it=400, noise=0.0, seed=0):
    r = np.random.default_rng(seed)
    for _ in range(it): x = x - lr*g(x) + noise*r.normal(size=len(x))
    return x
def newton_sad(x, it=30):
    Hs = lambda z: np.diag([2.0, -2.0 + 3*.25*z[1]**2])          # exact Hessian of sad_f
    for _ in range(it): x = x - np.linalg.solve(Hs(x), sad_g(x))
    return x
loc_f = lambda x: x**4 - 3*x**2 + x; loc_g = lambda x: 4*x**3 - 6*x + 1
R["landscape"] = dict(
    saddle_GD_from_exact_saddle=gd_path(sad_g, np.array([0., 0.])).round(4).tolist(),
    saddle_GD_with_noise=gd_path(sad_g, np.array([0., 0.]), noise=.01).round(3).tolist(),
    saddle_newton_from__03_05=newton_sad(np.array([0.3, 0.5])).round(4).tolist(),
    local_minima={str(x0): float(gd_path(lambda z: np.array([loc_g(z[0])]), np.array([x0]), lr=.01, it=2000)[0]) for x0 in [-2, -.5, .5, 2]},
    local_minima_values=dict(left=float(loc_f(-1.3)), right=float(loc_f(1.13))))
json.dump(R, open("../results/trackA_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
