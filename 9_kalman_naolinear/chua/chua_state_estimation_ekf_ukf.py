#!/usr/bin/env python3
"""
chua_state_estimation_ekf_ukf.py

Estimate v_C1 using EKF and UKF given only measurements of v_C2 and i_L.
Discretization: Ts = 0.01 s (10 ms).

Usage:
    python chua_state_estimation_ekf_ukf.py --file pcchua_dados.dat --plot --tune

Notes:
 - The script expects the data files to contain columns 'time', 'x', 'y', 'z'
   corresponding to time, vC1, vC2, iL respectively. Only 'y' and 'z' are used
   as measured signals.
 - Tuning is automated by optimizing two scalar scale factors:
      q_scale >= 0 multiplies a baseline Q0
      r_scale >= 0 multiplies a baseline R0
   Objective: minimize sum of squared autocorrelations of innovations (lags 1..K).
 - The user should inspect innovation autocorrelation plots and residuals.
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from collections.abc import Callable

# -------------------------
# Parameters (SI)
# -------------------------
PARAMS = {
    "C1": 30.14e-6,     # F
    "C2": 185.6e-6,     # F
    "L": 52.28,         # H
    "R": 1673.0,        # Ohm
    "R_L": 0.0,         # Ohm (default)
    "G_a": -0.801e-3,   # S
    "G_b": -0.365e-3,   # S
    "E": 1.74,          # V
    "d": 6.0,           # V (unused)
}

Ts = 0.01  # sampling time (s)

# -------------------------
# Utils: read data file
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
    # try to name columns sensibly if 10 cols as earlier convention
    if df.shape[1] >= 10:
        df.columns = ["time", "x", "y", "z", "xref", "yref", "zref", "ux", "uy", "uz"] + [f"c{i}" for i in range(10, df.shape[1])]
    elif df.shape[1] >= 4:
        df.columns = ["time", "x", "y", "z"] + [f"c{i}" for i in range(4, df.shape[1])]
    else:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df

# -------------------------
# Continuous-time model
# -------------------------
def G_of_v(v: np.ndarray, G_a: float, G_b: float, E: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    abs_v = np.abs(v)
    return np.where(abs_v < E, G_a, G_b + (G_a - G_b) * E / abs_v)

def f_continuous(t: float, state: np.ndarray, params: dict[str,float], r: np.ndarray | None=None) -> np.ndarray:
    """
    Continuous-time vector field for states [v1, v2, iL].
    Optional r = [r_x, r_y, r_z] external perturbations; if None, assume zero.
    """
    v1, v2, iL = state
    C1 = params["C1"]; C2 = params["C2"]; L = params["L"]; R = params["R"]; R_L = params["R_L"]
    G_a = params["G_a"]; G_b = params["G_b"]; E = params["E"]
    if r is None:
        rx = ry = rz = 0.0
    else:
        rx, ry, rz = r
    Gv = float(G_of_v(v1, G_a, G_b, E))
    dv1 = (1.0 / C1) * ( (v2 - v1) / R - Gv * v1 + rx )
    dv2 = (1.0 / C2) * ( (v1 - v2) / R + iL + ry )
    diL = (1.0 / L)  * ( -v2 + R_L * iL + rz )
    return np.array([dv1, dv2, diL], dtype=float)

# -------------------------
# Discretization: RK4 single step
# -------------------------
def rk4_step(x: np.ndarray, dt: float, f: Callable[[float, np.ndarray, dict], np.ndarray], t: float, params) -> np.ndarray:
    k1 = f(t, x, params)
    k2 = f(t + dt/2.0, x + dt * k1 / 2.0, params)
    k3 = f(t + dt/2.0, x + dt * k2 / 2.0, params)
    k4 = f(t + dt, x + dt * k3, params)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

def discrete_propagation(x: np.ndarray, dt: float, params: dict[str,float], r_func=None, t=0.0) -> np.ndarray:
    """
    Discrete-step state propagation: integrates continuous model from t to t+dt with RK4.
    r_func(t) -> [rx, ry, rz] if perturbations known, else None.
    We wrap f_continuous to include r at each call.
    """
    if r_func is None:
        fwrap = lambda tt, xx, p: f_continuous(tt, xx, p, r=None)
    else:
        fwrap = lambda tt, xx, p: f_continuous(tt, xx, p, r=np.asarray(r_func(tt)).reshape(3,))
    return rk4_step(x, dt, fwrap, t, params)

# -------------------------
# Jacobian of f w.r.t. state (continuous)
# We'll compute analytic partials where needed for EKF linearization.
# -------------------------
def jacobian_f(x: np.ndarray, params: dict[str,float]) -> np.ndarray:
    """
    Return A = df/dx (3x3) evaluated at state x (continuous-time Jacobian).
    Derivatives are computed analytically from the model equations.
    """
    v1, v2, iL = x
    C1 = params["C1"]; C2 = params["C2"]; L = params["L"]; R = params["R"]; R_L = params["R_L"]
    G_a = params["G_a"]; G_b = params["G_b"]; E = params["E"]

    # G(v1) piecewise. Compute derivative d(G(v1)*v1)/dv1 carefully.
    abs_v1 = abs(v1)
    if abs_v1 < E or abs_v1 == 0.0:
        Gv = G_a
        # derivative of G(v1) with respect to v1 is zero in linear region
        dGdv = 0.0
    else:
        Gv = G_b + (G_a - G_b) * E / abs_v1
        # derivative of G wrt v1: for v1>0: d/dv ( (G_a-G_b)*E/v1 ) = -(G_a-G_b)*E / v1^2
        # for v1<0 similar sign because abs_v1 = -v1 -> derivative is +(G_a-G_b)*E / v1^2 ? We'll compute using sign.
        sign = np.sign(v1)
        # We want dG/dv = -(G_a - G_b) * E * sign / (v1^2)
        # Derivation: abs_v = sign*v1. d(1/abs_v)/dv = -sign / v1^2
        dGdv = (G_a - G_b) * E * ( -np.sign(v1) ) / (v1*v1) if v1 != 0 else 0.0

    # derivative of term G(v1)*v1 -> d/dv1 [Gv*v1] = Gv + v1 * dGdv
    d_Gv_v1 = Gv + v1 * dGdv

    A = np.zeros((3,3), dtype=float)
    A[0,0] = (1.0 / C1) * ( -1.0 / params["R"] - d_Gv_v1 )  # dv1/dv1
    A[0,1] = (1.0 / C1) * ( 1.0 / params["R"] )           # dv1/dv2
    A[0,2] = 0.0                                          # dv1/diL

    A[1,0] = (1.0 / params["C2"]) * ( 1.0 / params["R"] ) # dv2/dv1
    A[1,1] = (1.0 / params["C2"]) * ( -1.0 / params["R"] )# dv2/dv2
    A[1,2] = (1.0 / params["C2"]) * 1.0                   # dv2/diL

    A[2,0] = 0.0
    A[2,1] = (-1.0 / params["L"])                         # diL/dv2
    A[2,2] = (1.0 / params["L"]) * params["R_L"]          # diL/diL

    return A

# -------------------------
# EKF implementation (discrete-time)
# We'll linearize continuous dynamics and use matrix exponential approx
# or simple Euler discretization of A: Ad = I + A*Ts (sufficient for small Ts).
# For higher accuracy one can compute matrix exponential. We implement both options.
# -------------------------
from scipy.linalg import expm

def discretize_A(A: np.ndarray, dt: float) -> np.ndarray:
    """Exact discretization for linear system xdot = A x -> x[k+1] = Ad x[k] using matrix exponential."""
    return expm(A * dt)

def ekf_run(z_time: np.ndarray, z_meas: np.ndarray, x0: np.ndarray, P0: np.ndarray,
            Q: np.ndarray, R: np.ndarray, params: dict[str,float], Ts: float,
            r_func=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    z_time: times array
    z_meas: measurements array shape (N,2) for [v2, iL]
    Returns arrays: x_est (N,3), P_trace (N,), innovations (N,2)
    """
    N = z_meas.shape[0]
    x_est = np.zeros((N, 3))
    P_store = np.zeros((N, 3, 3))
    innov = np.zeros((N, 2))
    x = x0.copy()
    P = P0.copy()
    t = z_time[0]

    H = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])  # measurement picks v2 and iL

    for k in range(N):
        tk = z_time[k]
        # propagate from t to t+Ts using discrete propagation on state and linearized A
        # compute A at current x
        A_cont = jacobian_f(x, params)
        Ad = discretize_A(A_cont, Ts)  # discrete A
        # non-linear propagation of the mean
        x = discrete_propagation(x, Ts, params, r_func, t)
        # propagate covariance: P = Ad P Ad^T + Qd
        # Here Q is already discrete-time process noise covariance supplied.
        P = Ad @ P @ Ad.T + Q

        # Measurement update at time tk+1 (we assume measurement is available at integer steps)
        z = z_meas[k]
        y_pred = H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        innovation = z - y_pred
        x = x + K @ innovation
        P = (np.eye(3) - K @ H) @ P

        x_est[k,:] = x
        P_store[k,:,:] = P
        innov[k,:] = innovation
        t += Ts

    return x_est, P_store, innov

# -------------------------
# UKF implementation
# -------------------------
def sigma_points(x: np.ndarray, P: np.ndarray, alpha=1e-3, beta=2.0, kappa=0.0):
    n = x.shape[0]
    lam = alpha**2 * (n + kappa) - n
    U = np.linalg.cholesky((n + lam) * P)
    sigmas = np.zeros((2*n+1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[1+i]   = x + U[:,i]
        sigmas[1+n+i] = x - U[:,i]
    # weights
    Wm = np.full(2*n+1, 1.0/(2*(n+lam)))
    Wc = Wm.copy()
    Wm[0] = lam/(n+lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    return sigmas, Wm, Wc

def ukf_run(z_time: np.ndarray, z_meas: np.ndarray, x0: np.ndarray, P0: np.ndarray,
            Q: np.ndarray, R: np.ndarray, params: dict[str,float], Ts: float,
            r_func=None, alpha=1e-3, beta=2.0, kappa=0.0):
    N = z_meas.shape[0]
    x_est = np.zeros((N,3))
    P_store = np.zeros((N,3,3))
    innov = np.zeros((N,2))
    x = x0.copy()
    P = P0.copy()
    H = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    for k in range(N):
        # 1) compute sigma points
        sigmas, Wm, Wc = sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)
        # 2) propagate sigma points through dynamics (discrete propagation)
        sigmas_prop = np.zeros_like(sigmas)
        tnow = z_time[k]  # propagate from tnow to tnow+Ts
        for i, s in enumerate(sigmas):
            sigmas_prop[i] = discrete_propagation(s, Ts, params, r_func, tnow)
        # 3) recover predicted mean and covariance
        x_pred = np.sum(Wm[:,None] * sigmas_prop, axis=0)
        P_pred = Q.copy()
        for i in range(sigmas.shape[0]):
            dx = sigmas_prop[i] - x_pred
            P_pred += Wc[i] * np.outer(dx, dx)

        # 4) predict measurements from sigma points
        z_sig = np.array([H @ sp for sp in sigmas_prop])
        z_pred = np.sum(Wm[:,None] * z_sig, axis=0)
        P_zz = R.copy()
        for i in range(sigmas.shape[0]):
            dz = z_sig[i] - z_pred
            P_zz += Wc[i] * np.outer(dz, dz)
        P_xz = np.zeros((3,2))
        for i in range(sigmas.shape[0]):
            dx = sigmas_prop[i] - x_pred
            dz = z_sig[i] - z_pred
            P_xz += Wc[i] * np.outer(dx, dz)

        # 5) update
        z = z_meas[k]
        S = P_zz
        K = P_xz @ np.linalg.inv(S)
        innov_k = z - z_pred
        x = x_pred + K @ innov_k
        P = P_pred - K @ S @ K.T

        x_est[k,:] = x
        P_store[k,:,:] = P
        innov[k,:] = innov_k

    return x_est, P_store, innov

# -------------------------
# Innovation autocorrelation objective and tuning
# -------------------------
def sample_autocorrelation(u: np.ndarray, nlags: int=10) -> np.ndarray:
    """
    Compute sample autocorrelation for lags 1..nlags for a 1-D sequence u.
    Returns array of autocorrelations for lags 1..nlags.
    """
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
                                  params: dict[str,float], Ts: float,
                                  r_func=None, nlags=10):
    """
    scales: [log10(q_scale), log10(r_scale)] to optimize over real line
    Returns scalar objective: sum of squared autocorrelations of all innovation channels and lags.
    Lower is better.
    """
    q_scale = 10.0**(scales[0])
    r_scale = 10.0**(scales[1])
    Q = Q0 * q_scale
    R = R0 * r_scale

    if filter_type == "EKF":
        x_est, P_store, innov = ekf_run(df_time, meas, x0, P0, Q, R, params, Ts, r_func)
    else:
        x_est, P_store, innov = ukf_run(df_time, meas, x0, P0, Q, R, params, Ts, r_func)
    # compute autocorrelations for each innovation channel
    obj = 0.0
    for ch in range(innov.shape[1]):
        ac = sample_autocorrelation(innov[:,ch], nlags=nlags)
        obj += np.sum(ac**2)
    # add small regularization to avoid degenerate Q->0 or R->0
    obj += 1e-12 * (q_scale + r_scale)
    return obj

def tune_scales(filter_type: str, df_time: np.ndarray, meas: np.ndarray,
                x0: np.ndarray, P0: np.ndarray, Q0: np.ndarray, R0: np.ndarray,
                params: dict[str,float], Ts: float, r_func=None):
    """
    Optimize log10(q_scale) and log10(r_scale) using Nelder-Mead on the objective.
    Start from 0 (scales = 1).
    Returns best q_scale, r_scale and final objective.
    """
    def obj_wrapped(s):
        return innovation_autocorr_objective(s, filter_type, df_time, meas, x0, P0, Q0, R0, params, Ts, r_func, nlags=10)

    x0_guess = np.array([0.0, 0.0])  # log10(q_scale)=0 => q_scale=1
    res = optimize.minimize(obj_wrapped, x0_guess, method='Nelder-Mead',
                            options={'maxiter':80, 'disp': True})
    best_scales = 10.0**(res.x)
    return best_scales[0], best_scales[1], res

# -------------------------
# Main: orchestration
# -------------------------
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="EKF and UKF state estimation for Chua-like system.")
    p.add_argument("--file", "-f", required=True, help="PCChua data file")
    p.add_argument("--plot", action="store_true", help="Show plots of results")
    p.add_argument("--tune", action="store_true", help="Run automatic tuning of Q/R scales (may be slow)")
    args = p.parse_args(argv)

    df = read_pcchua(args.file)
    # required columns: time, x (vC1), y (vC2), z (iL)
    if not {"time","x","y","z"}.issubset(df.columns):
        raise RuntimeError("File must contain columns 'time', 'x', 'y', 'z' named accordingly.")
    time = np.asarray(df["time"].values)
    x = np.asarray(df["x"].values)
    y = np.asarray(df["y"].values)
    z = np.asarray(df["z"].values)
    dt = Ts
    # Ensure time spacing matches Ts approximately. If not, we will resample onto uniform grid.
    dt_mean = np.mean(np.diff(time))
    if abs(dt_mean - Ts) > 1e-6:
        # resample with linear interpolation onto uniform grid spaced by Ts between first and last sample
        t_uniform = np.arange(time[0], time[-1] + 1e-9, Ts)
        interp_x = np.interp(t_uniform, time, x)
        interp_y = np.interp(t_uniform, time, y)
        interp_z = np.interp(t_uniform, time, z)
        time = t_uniform
        meas = np.vstack([interp_y, interp_z]).T  # measured y and z
    else:
        meas = np.vstack([y, z]).T

    N = meas.shape[0]

    # Baseline Q0 and R0 (discrete-time) choices:
    # We define Q0 to be small on states: here use diagonal with small entries scaled to state magnitudes.
    Q0 = np.diag([1e-6, 1e-6, 1e-8])  # baseline process noise cov (discrete-time)
    # Measurement noise baseline from sample variance of measurements
    meas_var = np.var(meas, axis=0, ddof=1)
    R0 = np.diag(np.maximum(meas_var * 1e-3, 1e-8))  # start with small measurement noise relative to variance

    # initial state and covariance: we only know measured v2 and iL; set initial v1 to 0 and moderate P
    x0 = np.array([0.0, meas[0,0], meas[0,1]])
    P0 = np.diag([1.0, 0.1, 0.01])

    # r_func: assume no external perturbation known, set to None
    r_func = None

    # Run initial EKF and UKF with baseline scales = 1
    print("Running baseline EKF...")
    Q_baseline = Q0.copy()
    R_baseline = R0.copy()
    x_ekf, P_ekf, innov_ekf = ekf_run(time, meas, x0, P0, Q_baseline, R_baseline, PARAMS, Ts, r_func)
    print("Running baseline UKF...")
    x_ukf, P_ukf, innov_ukf = ukf_run(time, meas, x0, P0, Q_baseline, R_baseline, PARAMS, Ts, r_func)

    # Optional tuning
    if args.tune:
        print("Tuning EKF Q/R scales to whiten innovations...")
        q_scale_ekf, r_scale_ekf, res_ekf = tune_scales("EKF", time, meas, x0, P0, Q0, R0, PARAMS, Ts, r_func)
        print("EKF best scales: q_scale = %.3g, r_scale = %.3g" % (q_scale_ekf, r_scale_ekf))
        Q_ekf = Q0 * q_scale_ekf
        R_ekf = R0 * r_scale_ekf
        x_ekf, P_ekf, innov_ekf = ekf_run(time, meas, x0, P0, Q_ekf, R_ekf, PARAMS, Ts, r_func)
        print("Tuning UKF Q/R scales to whiten innovations...")
        q_scale_ukf, r_scale_ukf, res_ukf = tune_scales("UKF", time, meas, x0, P0, Q0, R0, PARAMS, Ts, r_func)
        print("UKF best scales: q_scale = %.3g, r_scale = %.3g" % (q_scale_ukf, r_scale_ukf))
        Q_ukf = Q0 * q_scale_ukf
        R_ukf = R0 * r_scale_ukf
        x_ukf, P_ukf, innov_ukf = ukf_run(time, meas, x0, P0, Q_ukf, R_ukf, PARAMS, Ts, r_func)
    else:
        Q_ekf = Q0; R_ekf = R0; Q_ukf = Q0; R_ukf = R0

    # Diagnostics: innovation statistics and autocorrelations
    def diag_print(innov, name):
        print(f"---- {name} innovation statistics ----")
        for ch, label in enumerate(["v2", "iL"]):
            s = innov[:,ch]
            print(f"{label}: mean={np.mean(s):.3e}, var={np.var(s, ddof=1):.3e}")
            ac = sample_autocorrelation(s, nlags=10)
            print(f"{label} autocorr (lags1..5) = {ac[:5]}")
    diag_print(innov_ekf, "EKF")
    diag_print(innov_ukf, "UKF")

    # Plot results if requested
    if args.plot:
        t = time
        plt.figure(figsize=(12,9))
        plt.subplot(3,1,1)
        plt.plot(t, x_ekf[:,0], label='vC1_est (EKF)')
        plt.plot(t, x_ukf[:,0], label='vC1_est (UKF)', alpha=0.8)
        if 'x' in df.columns:
            # overlay true if present
            # resampled true x may be available in interpolation step
            try:
                # true vC1 on same time axis
                if np.mean(np.diff(time)) != Ts:
                    # earlier we resampled; the variable 'interp_x' may be unavailable; recompute
                    true_v1 = np.interp(t, time, x)
                else:
                    true_v1 = x[:len(t)]
                plt.plot(t, true_v1, label='vC1_true', linestyle='--', alpha=0.6)
            except Exception:
                pass
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

        # Autocorrelation plots
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

    # Save results to CSV
    out_df = pd.DataFrame({
        "time": time,
        "meas_v2": meas[:,0],
        "meas_iL": meas[:,1],
        "vC1_ekf": x_ekf[:,0],
        "vC2_ekf": x_ekf[:,1],
        "iL_ekf": x_ekf[:,2],
        "vC1_ukf": x_ukf[:,0],
        "vC2_ukf": x_ukf[:,1],
        "iL_ukf": x_ukf[:,2],
    })
    out_file = args.file + ".estimates.csv"
    out_df.to_csv(out_file, index=False)
    print("Estimates saved to:", out_file)
    print("Done.")

if __name__ == "__main__":
    main()
