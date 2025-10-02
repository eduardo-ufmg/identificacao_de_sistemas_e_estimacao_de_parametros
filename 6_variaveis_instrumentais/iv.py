import numpy as np

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
    # residuals
    res = Y - Phi @ theta
    sigma2 = np.var(res, ddof=1)
    return A, B, sigma2, start

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
    # initial LS
    Phi, Y, start = build_regression(y, u, na, nb, nk)
    theta_ls = np.linalg.pinv(Phi) @ Y
    # iterative IV using simulated outputs as instruments
    theta = theta_ls.copy()
    N = y.size
    for it in range(n_iter):
        y_sim = simulate_arx(theta, u, N, na, nb, nk)
        # Build instrument matrix Z using y_sim in place of y
        M = Phi.shape[0]
        Z = np.zeros_like(Phi)
        for idx, k in enumerate(range(start, N)):
            for i in range(1, na + 1):
                Z[idx, i - 1] = -safe_get(y_sim, k - i)
            for j in range(1, nb + 1):
                Z[idx, na + j - 1] = safe_get(u, k - nk - (j - 1))
        # IV estimator theta = (Z'Phi)^-1 Z'Y
        ZtPhi = Z.T @ Phi
        # Use pseudo-inverse in case of singularities
        theta = np.linalg.pinv(ZtPhi) @ (Z.T @ Y)
    # compute residual variance for the final theta
    res = Y - Phi @ theta
    sigma2 = np.var(res, ddof=1)
    return theta[:na].copy(), theta[na:].copy(), sigma2, start

if __name__ == "__main__":
    np.random.seed(0)
    n_steps = 1000

    # Generate random input signal
    u = np.random.uniform(-1, 1, n_steps)

    # Generate random noise signal
    noise = np.random.normal(0, 0.1, n_steps)

    A_true = np.array([-1.5, 0.7])
    B_true = np.array([1.0, 0.5])
    C_true = np.array([0.8, 0.0])
    F_true = A_true
    nk = 0

    # Preallocate output arrays
    armax_y = np.zeros(n_steps)
    oe_y = np.zeros(n_steps)

    # Simulate system outputs
    for k in range(n_steps):
        armax_y[k] = armax(A_true, B_true, C_true, nk, u, armax_y, noise, k)
        oe_y[k] = oe(F_true, B_true, nk, u, oe_y, noise, k)

    # Set model orders for estimation (true orders)
    na = 2
    nb = 2

    # --- ARX LS estimates ---
    A_ls_armax, B_ls_armax, s2_ls_armax, start_armax = estimate_arx_ls(armax_y, u, na, nb, nk)
    A_iv_armax, B_iv_armax, s2_iv_armax, _ = estimate_arx_iv(armax_y, u, na, nb, nk, n_iter=10)

    A_ls_oe, B_ls_oe, s2_ls_oe, start_oe = estimate_arx_ls(oe_y, u, na, nb, nk)
    A_iv_oe, B_iv_oe, s2_iv_oe, _ = estimate_arx_iv(oe_y, u, na, nb, nk, n_iter=10)

    # Print concise results
    print("ARMAX data (true A, B):", A_true.tolist(), B_true.tolist())
    print("LS estimate A:", A_ls_armax.tolist(), "B:", B_ls_armax.tolist(), "res_var:", float(s2_ls_armax))
    print("IV estimate A:", A_iv_armax.tolist(), "B:", B_iv_armax.tolist(), "res_var:", float(s2_iv_armax))
    print("")
    print("OE data (true F=A, B):", F_true.tolist(), B_true.tolist())
    print("LS estimate A:", A_ls_oe.tolist(), "B:", B_ls_oe.tolist(), "res_var:", float(s2_ls_oe))
    print("IV estimate A:", A_iv_oe.tolist(), "B:", B_iv_oe.tolist(), "res_var:", float(s2_iv_oe))