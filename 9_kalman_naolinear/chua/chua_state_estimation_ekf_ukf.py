"""

Features:
 - smooth nonlinearity G(v) to avoid singular derivative at v=0
 - augment measurement model with a slow bias on v2 (random-walk)
 - increase baseline process noise by a multiplier (default 10x)

Usage:
    python chua_state_estimation_ekf_ukf_improved.py --file pcchua_dados.dat --plot --tune

Flags:
    --no-smooth       disable smoothing of G(v)
    --no-bias         disable v2 bias augmentation
    --q-mult FLOAT    multiply baseline Q0 by this factor (default 10.0)
    --plot            show diagnostic plots
    --tune            run automatic tuning (optimizes two log-scales)
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from scipy.linalg import expm
from collections.abc import Callable

# -------------------------
# Parameters (SI)
# -------------------------
PARAMS = {
    "C1": 30.14e-6,
    "C2": 185.6e-6,
    "L": 52.28,
    "R": 1673.0,
    "R_L": 0.0,
    "G_a": -0.801e-3,
    "G_b": -0.365e-3,
    "E": 1.74,
}

Ts = 0.01  # sampling time

# -------------------------
# Utils: read data
# -------------------------
def read_pcchua(filename: str) -> pd.DataFrame:
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if not ln.strip().startswith('#') and ln.strip() != ""]
    if not data_lines:
        raise ValueError("No data lines found in file.")
    from io import StringIO
    s = "".join(data_lines)
    df = pd.read_csv(StringIO(s), delim_whitespace=True, header=None)
    if df.shape[1] >= 10:
        df.columns = ["time", "x", "y", "z", "xref", "yref", "zref", "ux", "uy", "uz"] + [f"c{i}" for i in range(10, df.shape[1])]
    elif df.shape[1] >= 4:
        df.columns = ["time", "x", "y", "z"] + [f"c{i}" for i in range(4, df.shape[1])]
    else:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df

# -------------------------
# Nonlinearity G(v) with optional smoothing
# -------------------------
def G_of_v(v: np.ndarray, G_a: float, G_b: float, E: float, smooth: bool = True) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if smooth:
        # avoid abs=0 and provide continuous derivative
        eps = 1e-6
        abs_v = np.sqrt(v**2 + eps)
        return G_b + (G_a - G_b) * E / abs_v
    else:
        abs_v = np.abs(v)
        return np.where(abs_v < E, G_a, G_b + (G_a - G_b) * E / abs_v)

# -------------------------
# Continuous-time model (supports optional bias state)
# -------------------------
def f_continuous(t: float, state: np.ndarray, params: dict, smooth: bool=True, augment_bias: bool=False, r: np.ndarray|None=None) -> np.ndarray:
    """
    state = [v1, v2, iL] or [v1, v2, iL, b] if augment_bias True.
    Measurement bias b is a slow random-walk: db/dt = 0
    """
    v1 = state[0]; v2 = state[1]; iL = state[2]
    if r is None:
        rx = ry = rz = 0.0
    else:
        rx, ry, rz = r
    C1 = params["C1"]; C2 = params["C2"]; L = params["L"]; R = params["R"]; R_L = params["R_L"]
    G_a = params["G_a"]; G_b = params["G_b"]; E = params["E"]

    Gv = float(G_of_v(v1, G_a, G_b, E, smooth=smooth))
    dv1 = (1.0 / C1) * ((v2 - v1) / R - Gv * v1 + rx)
    dv2 = (1.0 / C2) * ((v1 - v2) / R + iL + ry)
    diL = (1.0 / L) * (-v2 + R_L * iL + rz)

    if augment_bias:
        db = 0.0
        return np.array([dv1, dv2, diL, db], dtype=float)
    else:
        return np.array([dv1, dv2, diL], dtype=float)

# -------------------------
# Discretization RK4 step (works for variable state dims)
# -------------------------
def rk4_step(x: np.ndarray, dt: float, f: Callable, t: float, params, smooth: bool, augment_bias: bool, r_func=None) -> np.ndarray:
    def fwrap(tt, xx):
        r = None if r_func is None else r_func(tt)
        return f(tt, xx, params, smooth=smooth, augment_bias=augment_bias, r=r)
    k1 = fwrap(t, x)
    k2 = fwrap(t + dt/2.0, x + dt * k1 / 2.0)
    k3 = fwrap(t + dt/2.0, x + dt * k2 / 2.0)
    k4 = fwrap(t + dt, x + dt * k3)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

def discrete_propagation(x: np.ndarray, dt: float, params: dict, smooth: bool, augment_bias: bool, r_func=None, t=0.0) -> np.ndarray:
    return rk4_step(x, dt, f_continuous, t, params, smooth, augment_bias, r_func)

# -------------------------
# Jacobian of f w.r.t. state (continuous) with optional bias
# -------------------------
def jacobian_f(x: np.ndarray, params: dict, smooth: bool=True, augment_bias: bool=False) -> np.ndarray:
    v1 = x[0]; v2 = x[1]; iL = x[2]
    C1 = params["C1"]; C2 = params["C2"]; L = params["L"]; R = params["R"]; R_L = params["R_L"]
    G_a = params["G_a"]; G_b = params["G_b"]; E = params["E"]

    # smooth derivative: avoid singularities
    if smooth:
        eps = 1e-6
        abs_v1 = np.sqrt(v1**2 + eps)
        Gv = G_b + (G_a - G_b) * E / abs_v1
        # derivative of G wrt v1 using smooth abs_v1
        dGdv = (G_a - G_b) * (-E) * (v1) / (abs_v1**3)
    else:
        abs_v1 = abs(v1)
        if abs_v1 < E or abs_v1 == 0.0:
            Gv = G_a
            dGdv = 0.0
        else:
            Gv = G_b + (G_a - G_b) * E / abs_v1
            dGdv = (G_a - G_b) * E * ( -np.sign(v1) ) / (v1*v1) if v1 != 0 else 0.0

    d_Gv_v1 = Gv + v1 * dGdv

    n = 4 if augment_bias else 3
    A = np.zeros((n, n), dtype=float)
    A[0,0] = (1.0 / C1) * ( -1.0 / R - d_Gv_v1 )
    A[0,1] = (1.0 / C1) * ( 1.0 / R )
    A[0,2] = 0.0

    A[1,0] = (1.0 / C2) * ( 1.0 / R )
    A[1,1] = (1.0 / C2) * ( -1.0 / R )
    A[1,2] = (1.0 / C2) * 1.0

    A[2,0] = 0.0
    A[2,1] = (-1.0 / L)
    A[2,2] = (1.0 / L) * R_L

    if augment_bias:
        # bias state has zero dynamics derivative and does not affect others
        A[0,3] = 0.0
        A[1,3] = 0.0
        A[2,3] = 0.0
        A[3,3] = 0.0

    return A

# -------------------------
# Discretize A
# -------------------------
def discretize_A(A: np.ndarray, dt: float) -> np.ndarray:
    return expm(A * dt)

# -------------------------
# EKF & UKF (support variable state length)
# -------------------------
def ekf_run(z_time: np.ndarray, z_meas: np.ndarray, x0: np.ndarray, P0: np.ndarray,
            Q: np.ndarray, R: np.ndarray, params: dict, Ts: float,
            smooth: bool, augment_bias: bool, r_func=None):
    N = z_meas.shape[0]
    n = x0.size
    x_est = np.zeros((N, n))
    P_store = np.zeros((N, n, n))
    innov = np.zeros((N, z_meas.shape[1]))
    x = x0.copy()
    P = P0.copy()
    H = np.zeros((2, n))
    # measurement maps:
    # z[0] measures v2 plus bias if augment
    # z[1] measures iL
    H[0,1] = 1.0
    if augment_bias:
        H[0,3] = 1.0
    H[1,2] = 1.0

    t = z_time[0]
    for k in range(N):
        # linearize
        A_cont = jacobian_f(x, params, smooth=smooth, augment_bias=augment_bias)
        Ad = discretize_A(A_cont, Ts)
        x = discrete_propagation(x, Ts, params, smooth=smooth, augment_bias=augment_bias, r_func=r_func, t=t)
        P = Ad @ P @ Ad.T + Q

        z = z_meas[k]
        y_pred = H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.pinv(S)
        innovation = z - y_pred
        x = x + K @ innovation
        P = (np.eye(n) - K @ H) @ P

        x_est[k,:] = x
        P_store[k,:,:] = P
        innov[k,:] = innovation
        t += Ts

    return x_est, P_store, innov

def sigma_points(x: np.ndarray, P: np.ndarray, alpha=1e-3, beta=2.0, kappa=0.0):
    n = x.shape[0]
    lam = alpha**2 * (n + kappa) - n
    U = np.linalg.cholesky((n + lam) * P)
    sigmas = np.zeros((2*n+1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[1+i]   = x + U[:,i]
        sigmas[1+n+i] = x - U[:,i]
    Wm = np.full(2*n+1, 1.0/(2*(n+lam)))
    Wc = Wm.copy()
    Wm[0] = lam/(n+lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    return sigmas, Wm, Wc

def ukf_run(z_time: np.ndarray, z_meas: np.ndarray, x0: np.ndarray, P0: np.ndarray,
            Q: np.ndarray, R: np.ndarray, params: dict, Ts: float,
            smooth: bool, augment_bias: bool, r_func=None, alpha=1e-3, beta=2.0, kappa=0.0):
    N = z_meas.shape[0]
    n = x0.size
    x_est = np.zeros((N,n))
    P_store = np.zeros((N,n,n))
    innov = np.zeros((N,z_meas.shape[1]))
    x = x0.copy()
    P = P0.copy()
    H = np.zeros((2,n))
    H[0,1] = 1.0
    if augment_bias:
        H[0,3] = 1.0
    H[1,2] = 1.0

    for k in range(N):
        sigmas, Wm, Wc = sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)
        sigmas_prop = np.zeros_like(sigmas)
        tnow = z_time[k]
        for i, s in enumerate(sigmas):
            sigmas_prop[i] = discrete_propagation(s, Ts, params, smooth=smooth, augment_bias=augment_bias, r_func=r_func, t=tnow)
        x_pred = np.sum(Wm[:,None] * sigmas_prop, axis=0)
        P_pred = Q.copy()
        for i in range(sigmas.shape[0]):
            dx = sigmas_prop[i] - x_pred
            P_pred += Wc[i] * np.outer(dx, dx)

        z_sig = np.array([H @ sp for sp in sigmas_prop])
        z_pred = np.sum(Wm[:,None] * z_sig, axis=0)
        P_zz = R.copy()
        for i in range(sigmas.shape[0]):
            dz = z_sig[i] - z_pred
            P_zz += Wc[i] * np.outer(dz, dz)
        P_xz = np.zeros((n,2))
        for i in range(sigmas.shape[0]):
            dx = sigmas_prop[i] - x_pred
            dz = z_sig[i] - z_pred
            P_xz += Wc[i] * np.outer(dx, dz)

        z = z_meas[k]
        S = P_zz
        K = P_xz @ np.linalg.pinv(S)
        innov_k = z - z_pred
        x = x_pred + K @ innov_k
        P = P_pred - K @ S @ K.T

        x_est[k,:] = x
        P_store[k,:,:] = P
        innov[k,:] = innov_k

    return x_est, P_store, innov

# -------------------------
# Autocorrelation utilities and tuning objective
# -------------------------
def sample_autocorrelation(u: np.ndarray, nlags: int=10) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    u = u - np.mean(u)
    var = np.var(u, ddof=0)
    if var == 0:
        return np.zeros(nlags)
    N = u.size
    acf = np.array([np.dot(u[:N-l], u[l:]) / N / var for l in range(1, nlags+1)])
    return acf

def innovation_autocorr_objective(scales: np.ndarray, filter_type: str,
                                  df_time: np.ndarray, meas: np.ndarray,
                                  x0: np.ndarray, P0: np.ndarray,
                                  Q0: np.ndarray, R0: np.ndarray,
                                  params: dict, Ts: float,
                                  smooth: bool, augment_bias: bool, r_func=None, nlags=10):
    q_scale = 10.0**(scales[0])
    r_scale = 10.0**(scales[1])
    Q = Q0 * q_scale
    R = R0 * r_scale
    if filter_type == "EKF":
        x_est, P_store, innov = ekf_run(df_time, meas, x0, P0, Q, R, params, Ts, smooth, augment_bias, r_func)
    else:
        x_est, P_store, innov = ukf_run(df_time, meas, x0, P0, Q, R, params, Ts, smooth, augment_bias, r_func)
    obj = 0.0
    for ch in range(innov.shape[1]):
        ac = sample_autocorrelation(innov[:,ch], nlags=nlags)
        obj += np.sum(ac**2)
    obj += 1e-12 * (q_scale + r_scale)
    return obj

def tune_scales(filter_type: str, df_time: np.ndarray, meas: np.ndarray,
                x0: np.ndarray, P0: np.ndarray, Q0: np.ndarray, R0: np.ndarray,
                params: dict, Ts: float, smooth: bool, augment_bias: bool, r_func=None):
    def obj_wrapped(s):
        return innovation_autocorr_objective(s, filter_type, df_time, meas, x0, P0, Q0, R0, params, Ts, smooth, augment_bias, r_func, nlags=10)
    x0_guess = np.array([0.0, 0.0])
    res = optimize.minimize(obj_wrapped, x0_guess, method='Nelder-Mead', options={'maxiter':80, 'disp': True})
    best_scales = 10.0**(res.x)
    return best_scales[0], best_scales[1], res

# -------------------------
# Main
# -------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--file","-f", required=True)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--no-smooth", action="store_true", help="disable smooth G(v)")
    p.add_argument("--no-bias", action="store_true", help="disable bias augmentation")
    p.add_argument("--q-mult", type=float, default=10.0, help="multiply baseline Q0 by this factor (default 10)")
    args = p.parse_args(argv)

    df = read_pcchua(args.file)
    if not {"time","x","y","z"}.issubset(df.columns):
        raise RuntimeError("File must contain columns 'time','x','y','z'.")

    time = np.asarray(df["time"].values)
    true_v1 = np.asarray(df["x"].values)
    y = np.asarray(df["y"].values)
    z = np.asarray(df["z"].values)

    dt_mean = np.mean(np.diff(time))
    if abs(dt_mean - Ts) > 1e-6:
        t_uniform = np.arange(time[0], time[-1] + 1e-9, Ts)
        interp_x = np.interp(t_uniform, time, true_v1)
        interp_y = np.interp(t_uniform, time, y)
        interp_z = np.interp(t_uniform, time, z)
        time = t_uniform
        meas = np.vstack([interp_y, interp_z]).T
        true_v1 = interp_x
    else:
        meas = np.vstack([y, z]).T

    N = meas.shape[0]
    smooth = not args.no_smooth
    augment_bias = not args.no_bias
    q_mult = float(args.q_mult)

    # baseline Q0 and R0 (discrete-time)
    # If augment_bias, we add a small process noise on bias state (random walk)
    if augment_bias:
        Q0 = np.diag([1e-6, 1e-6, 1e-8, 1e-10])
    else:
        Q0 = np.diag([1e-6, 1e-6, 1e-8])
    Q0 = Q0 * q_mult

    meas_var = np.var(meas, axis=0, ddof=1)
    R0 = np.diag(np.maximum(meas_var * 1e-3, 1e-8))

    # initial state and covariance
    if augment_bias:
        x0 = np.array([0.0, meas[0,0], meas[0,1], 0.0])
        P0 = np.diag([1.0, 0.1, 0.01, 1e-4])
    else:
        x0 = np.array([0.0, meas[0,0], meas[0,1]])
        P0 = np.diag([1.0, 0.1, 0.01])

    r_func = None

    # Run baseline
    print("Running baseline EKF...")
    x_ekf, P_ekf, innov_ekf = ekf_run(time, meas, x0, P0, Q0, R0, PARAMS, Ts, smooth, augment_bias, r_func)
    print("Running baseline UKF...")
    x_ukf, P_ukf, innov_ukf = ukf_run(time, meas, x0, P0, Q0, R0, PARAMS, Ts, smooth, augment_bias, r_func)

    Q_ekf = Q0.copy(); R_ekf = R0.copy(); Q_ukf = Q0.copy(); R_ukf = R0.copy()

    if args.tune:
        print("Tuning EKF Q/R scales to whiten innovations...")
        q_scale_ekf, r_scale_ekf, res_ekf = tune_scales("EKF", time, meas, x0, P0, Q0, R0, PARAMS, Ts, smooth, augment_bias, r_func)
        Q_ekf = Q0 * q_scale_ekf
        R_ekf = R0 * r_scale_ekf
        print(f"EKF best scales: q_scale = {q_scale_ekf:.3g}, r_scale = {r_scale_ekf:.3g}")
        x_ekf, P_ekf, innov_ekf = ekf_run(time, meas, x0, P0, Q_ekf, R_ekf, PARAMS, Ts, smooth, augment_bias, r_func)

        print("Tuning UKF Q/R scales to whiten innovations...")
        q_scale_ukf, r_scale_ukf, res_ukf = tune_scales("UKF", time, meas, x0, P0, Q0, R0, PARAMS, Ts, smooth, augment_bias, r_func)
        Q_ukf = Q0 * q_scale_ukf
        R_ukf = R0 * r_scale_ukf
        print(f"UKF best scales: q_scale = {q_scale_ukf:.3g}, r_scale = {r_scale_ukf:.3g}")
        x_ukf, P_ukf, innov_ukf = ukf_run(time, meas, x0, P0, Q_ukf, R_ukf, PARAMS, Ts, smooth, augment_bias, r_func)

    # Diagnostics
    def diag_print(innov, name):
        print(f"---- {name} innovation statistics ----")
        for ch, label in enumerate(["v2", "iL"]):
            s = innov[:,ch]
            print(f"{label}: mean={np.mean(s):.3e}, var={np.var(s, ddof=1):.3e}")
            ac = sample_autocorrelation(s, nlags=10)
            print(f"{label} autocorr (lags1..5) = {ac[:5]}")
    diag_print(innov_ekf, "EKF")
    diag_print(innov_ukf, "UKF")

    # Save results
    out_df = {
        "time": time,
        "meas_v2": meas[:,0],
        "meas_iL": meas[:,1],
        "vC1_ekf": x_ekf[:,0],
        "vC2_ekf": x_ekf[:,1],
        "iL_ekf": x_ekf[:,2],
        "vC1_ukf": x_ukf[:,0],
        "vC2_ukf": x_ukf[:,1],
        "iL_ukf": x_ukf[:,2],
    }
    if augment_bias:
        out_df["bias_ekf"] = x_ekf[:,3]
        out_df["bias_ukf"] = x_ukf[:,3]
    if 'x' in df.columns:
        out_df["vC1_true"] = true_v1[:len(time)]

    out_file = args.file + ".estimates.csv"
    pd.DataFrame(out_df).to_csv(out_file, index=False)
    print("Estimates saved to:", out_file)

    # Plot if requested
    if args.plot:
        t = time
        plt.figure(figsize=(12,9))
        plt.subplot(3,1,1)
        plt.plot(t, x_ekf[:,0], label='vC1_est (EKF)')
        plt.plot(t, x_ukf[:,0], label='vC1_est (UKF)', alpha=0.8)
        if 'vC1_true' in out_df:
            plt.plot(t, out_df["vC1_true"], '--', label='vC1_true', alpha=0.6)
        plt.ylabel("v_C1 [V]"); plt.legend(); plt.title("Estimated v_C1")

        plt.subplot(3,1,2)
        plt.plot(t, innov_ekf[:,0], label='innov v2 (EKF)')
        plt.plot(t, innov_ukf[:,0], label='innov v2 (UKF)', alpha=0.8)
        plt.ylabel("innovation v2"); plt.legend()

        plt.subplot(3,1,3)
        plt.plot(t, innov_ekf[:,1], label='innov iL (EKF)')
        plt.plot(t, innov_ukf[:,1], label='innov iL (UKF)', alpha=0.8)
        plt.ylabel("innovation iL"); plt.legend()
        plt.xlabel("time [s]")
        plt.tight_layout()
        plt.show()

        # ACF plots
        def plot_acf_series(u, title):
            ac = sample_autocorrelation(u, nlags=50)
            plt.stem(range(1,len(ac)+1), ac)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.title(title)
            plt.xlabel("lag")
            plt.ylabel("autocorr")

        plt.figure(figsize=(10,6))
        plt.subplot(2,2,1)
        plot_acf_series(innov_ekf[:,0], "EKF innov v2 ACF")
        plt.subplot(2,2,2)
        plot_acf_series(innov_ekf[:,1], "EKF innov iL ACF")
        plt.subplot(2,2,3)
        plot_acf_series(innov_ukf[:,0], "UKF innov v2 ACF")
        plt.subplot(2,2,4)
        plot_acf_series(innov_ukf[:,1], "UKF innov iL ACF")
        plt.tight_layout()
        plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
