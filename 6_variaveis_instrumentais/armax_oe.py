import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------
# Core helper functions
# ---------------------
def safe_get(x: np.ndarray, idx: int) -> float:
    return float(x[idx]) if 0 <= idx < x.size else 0.0

def armax(A: np.ndarray, B: np.ndarray, C: np.ndarray, nk: int,
          u: np.ndarray, y: np.ndarray, noise: np.ndarray, k: int) -> float:
    na = 0 if A is None else A.size
    nb = 0 if B is None else B.size
    nc = 0 if C is None else C.size

    y_part = 0.0
    for i in range(1, na + 1):
        y_part -= A[i - 1] * safe_get(y, k - i)

    u_part = 0.0
    for j in range(1, nb + 1):
        u_idx = k - nk - (j - 1)
        u_part += B[j - 1] * safe_get(u, u_idx)

    e_part = safe_get(noise, k)
    for m in range(1, nc + 1):
        e_part += C[m - 1] * safe_get(noise, k - m)

    return float(y_part + u_part + e_part)

def oe(F: np.ndarray, B: np.ndarray, nk: int,
       u: np.ndarray, y: np.ndarray, noise: np.ndarray, k: int) -> float:
    nf = 0 if F is None else F.size
    nb = 0 if B is None else B.size

    y_part = 0.0
    for i in range(1, nf + 1):
        y_part -= F[i - 1] * safe_get(y, k - i)

    u_part = 0.0
    for j in range(1, nb + 1):
        u_idx = k - nk - (j - 1)
        u_part += B[j - 1] * safe_get(u, u_idx)

    e_part = safe_get(noise, k)
    for m in range(1, nf + 1):
        e_part += F[m - 1] * safe_get(noise, k - m)

    return float(y_part + u_part + e_part)

def build_regression(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int):
    start = max(na, nb + nk)
    N = y.size
    M = N - start
    Phi = np.zeros((M, na + nb))
    Y = np.zeros(M)
    for idx, k in enumerate(range(start, N)):
        for i in range(1, na + 1):
            Phi[idx, i - 1] = -safe_get(y, k - i)
        for j in range(1, nb + 1):
            Phi[idx, na + j - 1] = safe_get(u, k - nk - (j - 1))
        Y[idx] = safe_get(y, k)
    return Phi, Y, start

def estimate_arx_ls(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int):
    Phi, Y, start = build_regression(y, u, na, nb, nk)
    theta = np.linalg.pinv(Phi) @ Y
    A = theta[:na].copy()
    B = theta[na:].copy()
    res = Y - Phi @ theta
    sigma2 = np.var(res, ddof=1)
    return A, B, sigma2

def simulate_arx(theta: np.ndarray, u: np.ndarray, N: int, na: int, nb: int, nk: int):
    A = theta[:na]
    B = theta[na:]
    y = np.zeros(N)
    for k in range(N):
        y_part = 0.0
        for i in range(1, na + 1):
            y_part -= A[i - 1] * safe_get(y, k - i)
        u_part = 0.0
        for j in range(1, nb + 1):
            u_part += B[j - 1] * safe_get(u, k - nk - (j - 1))
        y[k] = y_part + u_part
    return y

def estimate_arx_iv(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int, n_iter: int = 5):
    Phi, Y, start = build_regression(y, u, na, nb, nk)
    theta_ls = np.linalg.pinv(Phi) @ Y
    theta = theta_ls.copy()
    N = y.size
    # iterative VI with simulated output as instrument
    for _ in range(n_iter):
        y_sim = simulate_arx(theta, u, N, na, nb, nk)
        M = Phi.shape[0]
        Z = np.zeros_like(Phi)
        for idx, k in enumerate(range(start, N)):
            for i in range(1, na + 1):
                Z[idx, i - 1] = -safe_get(y_sim, k - i)
            for j in range(1, nb + 1):
                Z[idx, na + j - 1] = safe_get(u, k - nk - (j - 1))
        ZtPhi = Z.T @ Phi
        theta = np.linalg.pinv(ZtPhi) @ (Z.T @ Y)
    res = Y - Phi @ theta
    sigma2 = np.var(res, ddof=1)
    return theta[:na].copy(), theta[na:].copy(), sigma2

# ---------------------
# Simulation parameters
# ---------------------
N_steps = 1000             # time series length per run
n_runs = 100               # number of Monte Carlo runs per generator
na = 2
nb = 2
nk = 0
n_iter_iv = 10            # iterations for the iterative VI estimator

# True system parameters (generator)
A_true = np.array([-1.5, 0.7])
B_true = np.array([1.0, 0.5])
C_true = np.array([0.8, 0.0])  # ARMAX noise filter (used in generator)
F_true = A_true                # OE generator uses F = A_true

# Input signal (fixed across runs for fairness)
rng = np.random.default_rng(0)
u_fixed = rng.uniform(-1, 1, N_steps)

# Pre-allocate storage for estimates
def zeros_storage():
    return {
        'A1_ls': np.zeros(n_runs), 'A2_ls': np.zeros(n_runs),
        'B1_ls': np.zeros(n_runs), 'B2_ls': np.zeros(n_runs),
        'A1_iv': np.zeros(n_runs), 'A2_iv': np.zeros(n_runs),
        'B1_iv': np.zeros(n_runs), 'B2_iv': np.zeros(n_runs),
    }

results_armax = zeros_storage()
results_oe = zeros_storage()

# ---------------------
# Monte Carlo simulations
# ---------------------
print("Starting Monte Carlo. Runs per generator:", n_runs)

for run in range(n_runs):
    # new white noise each run
    noise = rng.normal(0, 0.1, N_steps)

    # --- ARMAX generator simulation ---
    y_armax = np.zeros(N_steps)
    for k in range(N_steps):
        y_armax[k] = armax(A_true, B_true, C_true, nk, u_fixed, y_armax, noise, k)

    # Estimates on ARMAX data
    A_ls, B_ls, _ = estimate_arx_ls(y_armax, u_fixed, na, nb, nk)
    A_iv, B_iv, _ = estimate_arx_iv(y_armax, u_fixed, na, nb, nk, n_iter=n_iter_iv)

    results_armax['A1_ls'][run] = A_ls[0]
    results_armax['A2_ls'][run] = A_ls[1]
    results_armax['B1_ls'][run] = B_ls[0]
    results_armax['B2_ls'][run] = B_ls[1]

    results_armax['A1_iv'][run] = A_iv[0]
    results_armax['A2_iv'][run] = A_iv[1]
    results_armax['B1_iv'][run] = B_iv[0]
    results_armax['B2_iv'][run] = B_iv[1]

    # --- OE generator simulation ---
    noise2 = rng.normal(0, 0.1, N_steps)
    y_oe = np.zeros(N_steps)
    for k in range(N_steps):
        y_oe[k] = oe(F_true, B_true, nk, u_fixed, y_oe, noise2, k)

    # Estimates on OE data
    A_ls_o, B_ls_o, _ = estimate_arx_ls(y_oe, u_fixed, na, nb, nk)
    A_iv_o, B_iv_o, _ = estimate_arx_iv(y_oe, u_fixed, na, nb, nk, n_iter=n_iter_iv)

    results_oe['A1_ls'][run] = A_ls_o[0]
    results_oe['A2_ls'][run] = A_ls_o[1]
    results_oe['B1_ls'][run] = B_ls_o[0]
    results_oe['B2_ls'][run] = B_ls_o[1]

    results_oe['A1_iv'][run] = A_iv_o[0]
    results_oe['A2_iv'][run] = A_iv_o[1]
    results_oe['B1_iv'][run] = B_iv_o[0]
    results_oe['B2_iv'][run] = B_iv_o[1]

    if (run + 1) % 200 == 0:
        print(f"Completed runs: {run + 1}/{n_runs}")

print("Monte Carlo finished.")

# ---------------------
# Plotting helpers
# ---------------------
def plot_param_histograms(result_dict, model_name, out_dir="imagens", bins=60):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    params = [('A1', 'A1_ls', 'A1_iv'),
              ('A2', 'A2_ls', 'A2_iv'),
              ('B1', 'B1_ls', 'B1_iv'),
              ('B2', 'B2_ls', 'B2_iv')]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (label, key_ls, key_iv) in zip(axes, params):
        data_ls = result_dict[key_ls]
        data_iv = result_dict[key_iv]

        # determine common range and bins
        combined_min = min(data_ls.min(), data_iv.min())
        combined_max = max(data_ls.max(), data_iv.max())
        # add small margin
        span = combined_max - combined_min
        if span == 0:
            combined_min -= 0.5
            combined_max += 0.5
        else:
            combined_min -= 0.05 * span
            combined_max += 0.05 * span

        bins_edges = np.linspace(combined_min, combined_max, bins + 1)

        ax.hist(data_ls, bins=bins_edges, density=True, alpha=0.6, label='MQ', edgecolor='none')
        ax.hist(data_iv, bins=bins_edges, density=True, alpha=0.6, label='VI', edgecolor='none')
        ax.set_title(f"{model_name} {label}")
        ax.axvline(np.mean(data_ls), color='C0', linestyle='--', linewidth=1)
        ax.axvline(np.mean(data_iv), color='C1', linestyle='--', linewidth=1)
        ax.legend()
        ax.grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    out_path = Path(out_dir) / f"{model_name.lower()}_histogramas.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")

# ---------------------
# Create figures
# ---------------------
plot_param_histograms(results_armax, "ARMAX", out_dir="imagens", bins=60)
plot_param_histograms(results_oe, "OE", out_dir="imagens", bins=60)

# Print summary statistics
def print_summary(result_dict, model_name):
    print(f"\nSummary {model_name}:")
    for param in ['A1', 'A2', 'B1', 'B2']:
        ls = result_dict[f"{param}_ls"]
        iv = result_dict[f"{param}_iv"]
        print(f"{param}: MQ mean={ls.mean():.4f} std={ls.std(ddof=1):.4f} | VI mean={iv.mean():.4f} std={iv.std(ddof=1):.4f}")

print_summary(results_armax, "ARMAX")
print_summary(results_oe, "OE")

# End of script
