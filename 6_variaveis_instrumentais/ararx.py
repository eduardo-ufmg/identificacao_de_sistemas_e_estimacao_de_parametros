import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def safe_get(x: np.ndarray, idx: int) -> float:
    return float(x[idx]) if 0 <= idx < x.size else 0.0

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

def estimate_arx_ls_from_YPhi(Y: np.ndarray, Phi: np.ndarray):
    theta = np.linalg.pinv(Phi) @ Y
    return theta

def estimate_arx_ls(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int):
    Phi, Y, _ = build_regression(y, u, na, nb, nk)
    theta = estimate_arx_ls_from_YPhi(Y, Phi)
    return theta[:na].copy(), theta[na:].copy()

def simulate_generator(A: np.ndarray, B: np.ndarray, nk: int, u: np.ndarray, w: np.ndarray, H_alpha: float):
    N = u.size
    na = A.size
    nb = B.size
    y = np.zeros(N)
    e = np.zeros(N)
    for k in range(N):
        
        ar_part = 0.0
        for i in range(1, na + 1):
            ar_part -= A[i - 1] * safe_get(y, k - i)
        in_part = 0.0

        for j in range(1, nb + 1):
            in_part += B[j - 1] * safe_get(u, k - nk - (j - 1))

        e[k] = safe_get(w, k) - H_alpha * safe_get(e, k - 1)

        y[k] = ar_part + in_part + e[k]
    return y

def simulate_arx(theta: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int):
    A = theta[:na]
    B = theta[na:]
    N = u.size
    y = np.zeros(N)
    for k in range(N):
        ar_part = 0.0
        for i in range(1, na + 1):
            ar_part -= A[i - 1] * safe_get(y, k - i)
        in_part = 0.0
        for j in range(1, nb + 1):
            in_part += B[j - 1] * safe_get(u, k - nk - (j - 1))
        y[k] = ar_part + in_part
    return y

def iterative_iv(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int, n_iter: int = 8):
    Phi, Y, start = build_regression(y, u, na, nb, nk)
    theta = np.linalg.pinv(Phi) @ Y
    N = y.size
    for _ in range(n_iter):
        y_sim = simulate_arx(theta, u, na, nb, nk)
        M = Phi.shape[0]
        Z = np.zeros_like(Phi)
        for idx, k in enumerate(range(start, N)):
            for i in range(1, na + 1):
                Z[idx, i - 1] = -safe_get(y_sim, k - i)
            for j in range(1, nb + 1):
                Z[idx, na + j - 1] = safe_get(u, k - nk - (j - 1))
        ZtPhi = Z.T @ Phi
        theta = np.linalg.pinv(ZtPhi) @ (Z.T @ Y)
    return theta[:na].copy(), theta[na:].copy()

def prefilter_sequence(seq: np.ndarray, alpha: float):
    N = seq.size
    out = np.zeros_like(seq)
    for k in range(N):
        out[k] = seq[k] + alpha * safe_get(out, k - 1)
    return out

def estimate_els(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int, n_iter: int = 6):
    theta = np.zeros(na + nb)
    A_ls, B_ls = estimate_arx_ls(y, u, na, nb, nk)
    theta[:na] = A_ls
    theta[na:] = B_ls
    N = y.size
    for _ in range(n_iter):
        Phi, Y, _ = build_regression(y, u, na, nb, nk)
        res = Y - (Phi @ theta)
        if res.size < 2:
            alpha = 0.0
        else:
            num = np.dot(res[1:], res[:-1])
            den = np.dot(res[:-1], res[:-1])
            s = num / den if den != 0 else 0.0
            alpha = -s
        y_pref = prefilter_sequence(y, alpha)
        u_pref = prefilter_sequence(u, alpha)
        Phi_pref, Y_pref, _ = build_regression(y_pref, u_pref, na, nb, nk)
        theta = np.linalg.pinv(Phi_pref) @ Y_pref
    return theta[:na].copy(), theta[na:].copy()

def estimate_gls_with_true_filter(y: np.ndarray, u: np.ndarray, na: int, nb: int, nk: int, true_alpha: float):
    y_pref = prefilter_sequence(y, true_alpha)
    u_pref = prefilter_sequence(u, true_alpha)
    Phi_pref, Y_pref, _ = build_regression(y_pref, u_pref, na, nb, nk)
    theta = np.linalg.pinv(Phi_pref) @ Y_pref
    return theta[:na].copy(), theta[na:].copy()

N_steps = 50
n_runs = 100
na = 2
nb = 2
nk = 0
iv_iters = 4
els_iters = 4
true_alpha = 0.5
A_true = np.array([-1.5, 0.7])
B_true = np.array([1.0, 0.5])

rng = np.random.default_rng(1)
u_fixed = rng.uniform(-1, 1, N_steps)

def storage(n):
    return {
        'A1_els': np.zeros(n), 'A2_els': np.zeros(n), 'B1_els': np.zeros(n), 'B2_els': np.zeros(n),
        'A1_gls': np.zeros(n), 'A2_gls': np.zeros(n), 'B1_gls': np.zeros(n), 'B2_gls': np.zeros(n),
        'A1_iv':  np.zeros(n), 'A2_iv':  np.zeros(n), 'B1_iv':  np.zeros(n), 'B2_iv':  np.zeros(n),
    }

results = storage(n_runs)

for run in range(n_runs):
    w = rng.normal(0, 0.1, N_steps)
    y = simulate_generator(A_true, B_true, nk, u_fixed, w, H_alpha=true_alpha)

    A_els, B_els = estimate_els(y, u_fixed, na, nb, nk, n_iter=els_iters)
    A_gls, B_gls = estimate_gls_with_true_filter(y, u_fixed, na, nb, nk, true_alpha=true_alpha)
    A_iv, B_iv = iterative_iv(y, u_fixed, na, nb, nk, n_iter=iv_iters)

    results['A1_els'][run] = A_els[0]
    results['A2_els'][run] = A_els[1]
    results['B1_els'][run] = B_els[0]
    results['B2_els'][run] = B_els[1]

    results['A1_gls'][run] = A_gls[0]
    results['A2_gls'][run] = A_gls[1]
    results['B1_gls'][run] = B_gls[0]
    results['B2_gls'][run] = B_gls[1]

    results['A1_iv'][run] = A_iv[0]
    results['A2_iv'][run] = A_iv[1]
    results['B1_iv'][run] = B_iv[0]
    results['B2_iv'][run] = B_iv[1]

def plot_three_histograms(data_dict, outdir="imagens", bins=60):
    Path(outdir).mkdir(exist_ok=True, parents=True)
    params = [('A1', 'A1_els', 'A1_gls', 'A1_iv'),
              ('A2', 'A2_els', 'A2_gls', 'A2_iv'),
              ('B1', 'B1_els', 'B1_gls', 'B1_iv'),
              ('B2', 'B2_els', 'B2_gls', 'B2_iv')]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    labels = ['EMQ', 'GMQ', 'VI']
    colors = ['C0', 'C1', 'C2']

    for ax, (label, k_els, k_gls, k_iv) in zip(axes, params):
        d_els = data_dict[k_els]
        d_gls = data_dict[k_gls]
        d_iv = data_dict[k_iv]
        combined_min = min(d_els.min(), d_gls.min(), d_iv.min())
        combined_max = max(d_els.max(), d_gls.max(), d_iv.max())
        span = combined_max - combined_min
        if span == 0:
            combined_min -= 0.5
            combined_max += 0.5
        else:
            combined_min -= 0.05 * span
            combined_max += 0.05 * span
        bins_edges = np.linspace(combined_min, combined_max, bins + 1)

        ax.hist(d_els, bins=bins_edges, density=True, alpha=0.6, label='EMQ', color=colors[0], edgecolor='none')
        ax.hist(d_gls, bins=bins_edges, density=True, alpha=0.6, label='GMQ', color=colors[1], edgecolor='none')
        ax.hist(d_iv,  bins=bins_edges, density=True, alpha=0.6, label='VI',  color=colors[2], edgecolor='none')

        ax.set_title(label)
        ax.legend()
        ax.grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    outpath = Path(outdir) / "ararx_histogramas.png"
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {outpath}")

plot_three_histograms(results, outdir="imagens", bins=60)


