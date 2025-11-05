#!/usr/bin/env python3
"""
Usage examples:
    python chua_analysis.py --file pcchua_dados.dat --plot
    python chua_analysis.py --file pcchua_pert.dat --fit-rl
    python chua_analysis.py --file pcchua_dados.dat --simulate --tmax 5.0

What it does:
 - Parses PCChua-style data files (skips '#' header lines).
 - Implements G(v) piecewise law.
 - Builds RHS of the three-state model:
       dv1/dt = (1/C1) * { (v2 - v1)/R - G(v1)*v1 + r_x(t) }
       dv2/dt = (1/C2) * { (v1 - v2)/R + iL + r_y(t) }
       diL/dt = (1/L)  * { -v2 + R_L * iL + r_z(t) }
 - Computes measured derivatives from data by centered finite difference.
 - Computes estimates of r_x,r_y,r_z from measured states+derivatives.
 - Optionally compares estimates to recorded ux,uy,uz in the file.
 - Can simulate model given initial condition and (optionally) external r(t).
 - Can fit R_L by minimizing difference between model-derived r_z and measured uz.

Dependencies:
    numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy import interpolate, integrate, optimize
import matplotlib.pyplot as plt
import sys

# ---------------------------
# Default parameters (SI)
# ---------------------------
DEFAULTS: dict[str, float] = {
    "C1": 30.14e-6,     # F
    "C2": 185.6e-6,     # F
    "L": 52.28,         # H
    "R": 1673.0,        # Ohm
    "R_L": 0.0,         # Ohm (can be fitted)
    "G_a": -0.801e-3,   # S
    "G_b": -0.365e-3,   # S
    "E": 1.74,          # V
    "d": 6.0,           # V (unused by model but kept)
}

# ---------------------------
# File parsing
# ---------------------------
def read_pcchua(filename: str) -> pd.DataFrame:
    """
    Read PCChua-style file. Skips leading '#' lines.
    Expected column layout (10 columns):
      time, x, y, z, xref, yref, zref, ux, uy, uz
    Returns pandas DataFrame with named columns if possible.
    """
    # skip comment lines manually
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if not ln.strip().startswith('#') and ln.strip() != ""]
    if not data_lines:
        raise ValueError("No data lines found in file: " + filename)

    # Count columns on first data line
    first = data_lines[0].strip().split()
    ncols = len(first)
    # Build default names based on ncols
    if ncols >= 10:
        names = ["time", "x", "y", "z", "xref", "yref", "zref", "ux", "uy", "uz"] + [f"c{i}" for i in range(10, ncols)]
    elif ncols == 7:
        names = ["time", "x", "y", "z", "xref", "yref", "zref"]
    else:
        names = [f"c{i}" for i in range(ncols)]
    from io import StringIO
    s = "".join(data_lines)
    df = pd.read_csv(StringIO(s), delim_whitespace=True, header=None, names=names, comment='#')
    return df

# ---------------------------
# Piecewise G(v)
# ---------------------------
def G_of_v(v: np.ndarray, G_a: float, G_b: float, E: float) -> np.ndarray:
    """
    Piecewise conductance:
      if |v| < E: return G_a
      else: return G_b + (G_a - G_b)*E/|v|
    Works vectorized on numpy arrays or scalars.
    """
    v = np.asarray(v, dtype=float)
    abs_v = np.abs(v)
    out = np.where(abs_v < E, G_a, G_b + (G_a - G_b) * E / abs_v)
    return out

# ---------------------------
# RHS of the system
# ---------------------------
def make_u_interpolators(time: np.ndarray, ux: np.ndarray | None, uy: np.ndarray | None, uz: np.ndarray | None):
    """
    Returns interpolator functions for r_x(t), r_y(t), r_z(t).
    If any channel is None, that interpolator returns 0.
    """
    if ux is None:
        interp_x = lambda t: 0.0 if np.isscalar(t) else np.zeros_like(np.atleast_1d(t))
    else:
        f = interpolate.interp1d(time, ux, kind='linear', bounds_error=False)
        interp_x = lambda t: f(t)

    if uy is None:
        interp_y = lambda t: 0.0 if np.isscalar(t) else np.zeros_like(np.atleast_1d(t))
    else:
        f = interpolate.interp1d(time, uy, kind='linear', bounds_error=False)
        interp_y = lambda t: f(t)

    if uz is None:
        interp_z = lambda t: 0.0 if np.isscalar(t) else np.zeros_like(np.atleast_1d(t))
    else:
        f = interpolate.interp1d(time, uz, kind='linear', bounds_error=False)
        interp_z = lambda t: f(t)

    return interp_x, interp_y, interp_z

def rhs(t: float, state: np.ndarray, params: dict[str, float], rx_fun, ry_fun, rz_fun) -> np.ndarray:
    """
    Compute derivatives for given state [v1, v2, iL] at time t.
    rx_fun, ry_fun, rz_fun are callable functions of t returning the r signals.
    """
    v1, v2, iL = state
    C1 = params["C1"]; C2 = params["C2"]; L = params["L"]; R = params["R"]; R_L = params["R_L"]
    G_a = params["G_a"]; G_b = params["G_b"]; E = params["E"]

    Gv = G_of_v(v1, G_a, G_b, E)

    rx = float(np.atleast_1d(rx_fun(t))[0])
    ry = float(np.atleast_1d(ry_fun(t))[0])
    rz = float(np.atleast_1d(rz_fun(t))[0])

    dv1 = (1.0 / C1) * ( (v2 - v1) / R - Gv * v1 + rx )
    dv2 = (1.0 / C2) * ( (v1 - v2) / R + iL + ry )
    diL = (1.0 / L)  * ( -v2 + R_L * iL + rz )

    return np.array([dv1, dv2, diL], dtype=float)

# ---------------------------
# Measured derivative & residuals
# ---------------------------
def measured_derivatives(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """
    Compute time derivatives using centered differences.
    Returns a DataFrame with columns dx_dt, dy_dt, dz_dt.
    """
    t = np.asarray(df[time_col].values)
    # use numpy.gradient which handles non-uniform spacing
    dx = np.gradient(np.asarray(df["x"].values), t)
    dy = np.gradient(np.asarray(df["y"].values), t)
    dz = np.gradient(np.asarray(df["z"].values), t)
    return pd.DataFrame({"time": t, "dx_dt": dx, "dy_dt": dy, "dz_dt": dz})

def compute_r_estimates(df: pd.DataFrame, derivatives: pd.DataFrame, params: dict[str,float]) -> pd.DataFrame:
    """
    From measured states and derivatives compute r_x, r_y, r_z estimates using algebraic rearrangements:
      r_x = C1*dx_dt - ( (v2 - v1)/R - G(v1)*v1 )
      r_y = C2*dy_dt - ( (v1 - v2)/R + iL )
      r_z = L*dz_dt + v2 - R_L * iL
    Returns DataFrame with columns time, rx_est, ry_est, rz_est
    """
    v1 = np.asarray(df["x"].values)
    v2 = np.asarray(df["y"].values)
    iL = np.asarray(df["z"].values)
    dx = np.asarray(derivatives["dx_dt"].values)
    dy = np.asarray(derivatives["dy_dt"].values)
    dz = np.asarray(derivatives["dz_dt"].values)

    C1 = params["C1"]; C2 = params["C2"]; L = params["L"]; R = params["R"]; R_L = params["R_L"]
    Gv = G_of_v(v1, params["G_a"], params["G_b"], params["E"])

    rx = C1 * dx - ( (v2 - v1) / R - Gv * v1 )
    ry = C2 * dy - ( (v1 - v2) / R + iL )
    rz = L * dz + v2 - R_L * iL

    return pd.DataFrame({"time": df["time"].values, "rx_est": rx, "ry_est": ry, "rz_est": rz})

# ---------------------------
# Simulation helper
# ---------------------------
def simulate(initial_state: np.ndarray, params: dict[str,float], rx_fun, ry_fun, rz_fun,
             t_span: tuple[float,float], t_eval: np.ndarray | None = None, rtol=1e-8, atol=1e-10):
    """
    Simulate using solve_ivp, returns solution object.
    """
    fun = lambda t, y: rhs(t, y, params, rx_fun, ry_fun, rz_fun)
    sol = integrate.solve_ivp(fun, t_span, initial_state, t_eval=t_eval, rtol=rtol, atol=atol, method='RK45')
    return sol

# ---------------------------
# Parameter fitting (example: fit R_L)
# ---------------------------
def fit_RL(df: pd.DataFrame, derivatives: pd.DataFrame, params: dict[str,float], measured_uz: np.ndarray | None = None):
    """
    Fit R_L by minimizing (rz_est - uz_meas)^2 if uz present.
    If uz not present, fit by minimizing variance or regularization on rz_est (less meaningful).
    Returns OptimizeResult and best-fit params copy.
    """
    # target: recorded uz if available, else zeros
    t = df["time"].values
    if measured_uz is not None:
        uz_target = measured_uz
    else:
        uz_target = np.zeros_like(t)

    def objective(x):
        R_L_candidate = x[0]
        p = params.copy()
        p["R_L"] = float(R_L_candidate)
        rz_df = compute_r_estimates(df, derivatives, p)["rz_est"].values
        res = rz_df - uz_target
        # return residuals (vector) for least_squares
        return res

    x0 = np.array([params.get("R_L", 0.0)])
    res = optimize.least_squares(objective, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12, verbose=2)
    best = params.copy()
    best["R_L"] = float(res.x[0])
    return res, best

# ---------------------------
# Plotting
# ---------------------------
def quick_plot(df: pd.DataFrame, derivatives: pd.DataFrame, r_est: pd.DataFrame):
    t = df["time"].values
    fig, axs = plt.subplots(4,1, figsize=(10,10), sharex=True)
    axs[0].plot(t, df["x"], label='vC1 (x)')
    axs[0].plot(t, df["y"], label='vC2 (y)')
    axs[0].plot(t, df["z"], label='iL (z)')
    axs[0].legend()
    axs[0].set_ylabel("states")

    axs[1].plot(t, derivatives["dx_dt"], label='dx/dt')
    axs[1].plot(t, derivatives["dy_dt"], label='dy/dt')
    axs[1].plot(t, derivatives["dz_dt"], label='dz/dt')
    axs[1].legend()
    axs[1].set_ylabel("derivatives")

    axs[2].plot(t, r_est["rx_est"], label='r_x est')
    if "ux" in df.columns:
        axs[2].plot(t, df["ux"], label='ux (meas)')
    axs[2].legend()
    axs[2].set_ylabel("r_x")

    axs[3].plot(t, r_est["ry_est"], label='r_y est')
    axs[3].plot(t, r_est["rz_est"], label='r_z est')
    if "uy" in df.columns:
        axs[3].plot(t, df["uy"], label='uy (meas)')
    if "uz" in df.columns:
        axs[3].plot(t, df["uz"], label='uz (meas)')
    axs[3].legend()
    axs[3].set_ylabel("r_y, r_z")
    axs[3].set_xlabel("time [s]")

    plt.tight_layout()
    plt.show()

# ---------------------------
# Main / CLI
# ---------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Chua-like system analysis and simulation script.")
    p.add_argument("--file", "-f", required=True, help="PCChua data file (pcchua_dados.dat or pcchua_pert.dat)")
    p.add_argument("--plot", action="store_true", help="Produce quick diagnostic plots")
    p.add_argument("--simulate", action="store_true", help="Simulate model forward using recorded rx,ry,rz as inputs")
    p.add_argument("--tmax", type=float, default=None, help="Simulation stop time (s). Default: last data time")
    p.add_argument("--fit-rl", action="store_true", help="Fit R_L parameter using least-squares to measured uz (if present)")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful in non-interactive contexts)")
    args = p.parse_args(argv)

    # Read data
    df = read_pcchua(args.file)
    # Validate required columns
    if "time" not in df.columns:
        raise RuntimeError("File read but no 'time' column found.")
    # If ux/uy/uz not present they will be handled
    time = np.asarray(df["time"].values)
    ux = np.asarray(df["ux"].values) if "ux" in df.columns else None
    uy = np.asarray(df["uy"].values) if "uy" in df.columns else None
    uz = np.asarray(df["uz"].values) if "uz" in df.columns else None

    params = DEFAULTS.copy()

    # compute derivatives
    deriv = measured_derivatives(df, time_col="time")
    # compute r estimates
    r_est = compute_r_estimates(df, deriv, params)

    # make interpolators for external r signals using measured channels if present
    rx_fun, ry_fun, rz_fun = make_u_interpolators(time, ux, uy, uz)

    # Optionally fit R_L using uz channel if present
    if args.fit_rl:
        print("Fitting R_L ...")
        if uz is None:
            print("Warning: uz not present in file. Fit will minimize rz_est to zero. Consider using file with measured uz.")
        res, best_params = fit_RL(df, deriv, params, measured_uz=uz)
        print("Fit result:", res.message)
        print("Best-fit R_L:", best_params["R_L"])
        params = best_params

    # Optionally simulate forward using the recorded rx,ry,rz as inputs
    if args.simulate:
        t0 = float(time[0])
        tmax = args.tmax if args.tmax is not None else float(time[-1])
        t_eval = np.linspace(t0, tmax, max(2000, len(time)))
        init_state = np.array([float(df["x"].iloc[0]), float(df["y"].iloc[0]), float(df["z"].iloc[0])])
        sol = simulate(init_state, params, rx_fun, ry_fun, rz_fun, (t0, tmax), t_eval=t_eval)
        print(f"Simulation finished. nsteps={sol.y.shape[1]}; success={sol.success}")

        # Add simulated states to a small DataFrame for convenience
        sim_df = pd.DataFrame({"time": sol.t, "x_sim": sol.y[0], "y_sim": sol.y[1], "z_sim": sol.y[2]})
        # simple plot comparing measured vs simulated if requested
        if args.plot:
            plt.figure(figsize=(8,6))
            plt.subplot(3,1,1)
            plt.plot(df["time"], df["x"], label="x_meas")
            plt.plot(sim_df["time"], sim_df["x_sim"], label="x_sim", alpha=0.8)
            plt.legend(); plt.ylabel("vC1 (x)")
            plt.subplot(3,1,2)
            plt.plot(df["time"], df["y"], label="y_meas")
            plt.plot(sim_df["time"], sim_df["y_sim"], label="y_sim", alpha=0.8)
            plt.legend(); plt.ylabel("vC2 (y)")
            plt.subplot(3,1,3)
            plt.plot(df["time"], df["z"], label="z_meas")
            plt.plot(sim_df["time"], sim_df["z_sim"], label="z_sim", alpha=0.8)
            plt.legend(); plt.ylabel("iL (z)"); plt.xlabel("time [s]")
            plt.tight_layout()
            if not args.no_show:
                plt.show()

    # plotting of measured derivatives and r estimates
    if args.plot:
        quick_plot(df, deriv, r_est)
        if not args.no_show:
            plt.show()

    # Save computed estimates to CSV for offline processing
    out_df = df.copy()
    out_df = out_df.assign(dx_dt = deriv["dx_dt"].values,
                           dy_dt = deriv["dy_dt"].values,
                           dz_dt = deriv["dz_dt"].values,
                           rx_est = r_est["rx_est"].values,
                           ry_est = r_est["ry_est"].values,
                           rz_est = r_est["rz_est"].values)
    out_file = args.file + ".processed.csv"
    out_df.to_csv(out_file, index=False)
    print("Processed data saved to:", out_file)
    print("Done.")

if __name__ == "__main__":
    main()
