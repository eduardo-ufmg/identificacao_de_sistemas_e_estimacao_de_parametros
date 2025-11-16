import argparse
import os
from itertools import combinations_with_replacement
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import Ridge


def read_data(fname, input_col=0, output_col=-1, skip_header=False):
    data = np.loadtxt(fname)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    u = data[:, input_col].astype(float)
    y = data[:, output_col].astype(float)
    return u, y


def laguerre_filter_coeffs(a, order):
    """
    Return numerator (b) and denominator (a_coeffs) for Laguerre basis transfer:
       L_order(z) = sqrt(1-a^2) * (z^-1 - a)^order / (1 - a z^-1)^(order+1)
    Coeff arrays are in z^-1 polynomial order:
       b[k] corresponds to coefficient for z^-k (k=0..order)
       a_coeffs[k] corresponds to coefficient for z^-k (k=0..order+1)
    """
    scale = np.sqrt(1 - a**2)
    # numerator coefficients b[k] for z^-k: comb(order, k) * (-a)^(order-k)
    b = np.array(
        [comb(order, k) * ((-a) ** (order - k)) for k in range(order + 1)], dtype=float
    )
    b = scale * b
    # denominator coefficients a_coeffs[k] for z^-k: comb(order+1, k) * (-a)^k
    a_coeffs = np.array(
        [comb(order + 1, k) * ((-a) ** k) for k in range(order + 2)], dtype=float
    )
    return b, a_coeffs


def compute_laguerre_filtered_signals(u, a, M):
    """
    Compute M Laguerre-filtered signals s[:, i] = L_i(z) * u.
    Returns array shape (N, M).
    """
    N = len(u)
    s = np.zeros((N, M), dtype=float)
    for i in range(M):
        b, a_coeffs = laguerre_filter_coeffs(a, i)
        # lfilter expects arrays b (len i+1) and a_coeffs (len i+2)
        s[:, i] = lfilter(b, a_coeffs, u)
    return s


def build_design_matrix(s, order):
    """
    Build design matrix X from Laguerre-filtered signals s (N x M)
    order: 1, 2, or 3 (includes all lower orders)
    Returns X (N x P) and a dictionary with indices breakdown for interpretation.
    """
    N, M = s.shape
    cols = []
    info = {"M": M, "order_breakdown": {}}
    idx = 0
    # first order
    cols.append(s)  # shape N x M
    info["order_breakdown"]["first"] = (idx, idx + M)
    idx += M
    # second order
    if order >= 2:
        combos2 = list(combinations_with_replacement(range(M), 2))
        X2 = np.empty((N, len(combos2)), dtype=float)
        for k, (i, j) in enumerate(combos2):
            X2[:, k] = s[:, i] * s[:, j]
        cols.append(X2)
        info["order_breakdown"]["second"] = (idx, idx + X2.shape[1])
        info["second_combos"] = combos2
        idx += X2.shape[1]
    # third order
    if order >= 3:
        combos3 = list(combinations_with_replacement(range(M), 3))
        X3 = np.empty((N, len(combos3)), dtype=float)
        for k, (i, j, k3) in enumerate(combos3):
            X3[:, k] = s[:, i] * s[:, j] * s[:, k3]
        cols.append(X3)
        info["order_breakdown"]["third"] = (idx, idx + X3.shape[1])
        info["third_combos"] = combos3
        idx += X3.shape[1]
    X = np.concatenate(cols, axis=1)
    return X, info


def fit_ridge(X, y, alpha=1e-6):
    """
    Fit Ridge regression and return trained model object (sklearn Ridge).
    """
    model = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
    model.fit(X, y)
    return model


def predict_from_model(model, s, order, info):
    """
    Build design matrix for s and use model to predict y_hat.
    """
    X, _ = build_design_matrix(s, order)
    return model.predict(X)


def reconstruct_kernels_from_coeffs(model_coef, a, M, order, impulse_len):
    """
    Reconstruct approximated impulse responses of kernels using Laguerre impulse responses.
    Returns:
     - h1: vector length impulse_len
     - h2: array (impulse_len, impulse_len) or None
     - h3: array (impulse_len, impulse_len, impulse_len) or None
    """
    # compute Laguerre impulse responses l_i[k] by filtering delta
    delta = np.zeros(impulse_len, dtype=float)
    delta[0] = 1.0
    lag_imp = np.zeros((M, impulse_len), dtype=float)
    for i in range(M):
        b, a_coeffs = laguerre_filter_coeffs(a, i)
        lag_imp[i, :] = lfilter(b, a_coeffs, delta)

    # coefficients ordering: model.coef_ corresponds to columns in X produced by build_design_matrix
    coef = model_coef.copy()
    ptr = 0
    h1 = None
    h2 = None
    h3 = None
    # first order
    c1 = coef[ptr : ptr + M]
    ptr += M
    h1 = np.tensordot(c1, lag_imp, axes=(0, 0))  # sum_i c1_i * l_i[k] -> (impulse_len,)
    # second order
    if order >= 2:
        combos2 = list(combinations_with_replacement(range(M), 2))
        c2 = coef[ptr : ptr + len(combos2)]
        ptr += len(combos2)
        # build h2 as sum_{p} c2_p * outer(l_i, l_j)
        h2 = np.zeros((impulse_len, impulse_len), dtype=float)
        for coeff, (i, j) in zip(c2, combos2):
            h2 += coeff * np.outer(lag_imp[i, :], lag_imp[j, :])
    if order >= 3:
        combos3 = list(combinations_with_replacement(range(M), 3))
        c3 = coef[ptr : ptr + len(combos3)]
        ptr += len(combos3)
        # build h3 -- careful memory; if impulse_len is large this will be heavy
        h3 = np.zeros((impulse_len, impulse_len, impulse_len), dtype=float)
        for coeff, (i, j, k) in zip(c3, combos3):
            # outer(outer(l_i, l_j), l_k) -> broadcast
            h3 += coeff * np.einsum(
                "a,b,c->abc", lag_imp[i, :], lag_imp[j, :], lag_imp[k, :]
            )
    return h1, h2, h3


def metrics(y, yhat):
    err = y - yhat
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    var_y = np.var(y)
    nmse = mse / var_y if var_y != 0 else np.inf
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {"RMSE": rmse, "NMSE": nmse, "R2": r2}


def save_model(fn_out, model, a, M, order, info):
    """
    Save model to npz file including coefficients and meta.
    """
    meta = {
        "a": a,
        "M": M,
        "order": order,
        "info": info,
        "coef": model.coef_,
        "intercept": model.intercept_,
    }
    np.savez(fn_out, **meta)


def load_model(fn_npz):
    d = np.load(fn_npz, allow_pickle=True)
    return dict(d)


def main():
    parser = argparse.ArgumentParser(
        description="Volterra-Laguerre identification (orders 1-3)."
    )
    parser.add_argument(
        "file",
        help="Input data file (.txt) with columns; first column is u, last column is output t4.",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.6,
        help="Laguerre pole parameter (0 < a < 1). Default 0.6.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=6,
        help="Number of Laguerre basis functions (M). Default 6.",
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Highest Volterra order. Default 3.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Ridge regularization alpha. Default 1e-3.",
    )
    parser.add_argument(
        "--impulse-length",
        type=int,
        default=128,
        help="Impulse length for kernel reconstruction. Default 128.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="volterra_model.npz",
        help="Filename to save fitted model (npz).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save model to disk."
    )
    parser.add_argument(
        "--no-plot-compare",
        action="store_true",
        help="Do not plot comparison of output and prediction.",
    )
    parser.add_argument(
        "--input-col",
        type=int,
        default=0,
        help="Index of input column in file. Default 0.",
    )
    parser.add_argument(
        "--output-col",
        type=int,
        default=-1,
        help="Index of output column in file. Default -1 (last).",
    )
    args = parser.parse_args()

    if not (0.0 < args.a < 1.0):
        raise ValueError("Parameter a must satisfy 0 < a < 1.")

    u, y = read_data(args.file, input_col=args.input_col, output_col=args.output_col)
    N = len(u)
    if N != len(y):
        raise ValueError("Input and output length mismatch.")

    # compute Laguerre filtered signals
    s = compute_laguerre_filtered_signals(u, args.a, args.M)

    # build design matrix
    X, info = build_design_matrix(s, args.order)
    info["design_columns"] = X.shape[1]

    # fit ridge
    model = fit_ridge(X, y, alpha=args.alpha)

    # predictions and metrics on training data
    yhat = model.predict(X)
    m = metrics(y, yhat)

    # reconstruct kernels
    h1, h2, h3 = reconstruct_kernels_from_coeffs(
        model.coef_, args.a, args.M, args.order, args.impulse_length
    )

    # print concise report
    print("Volterra-Laguerre identification report")
    print(f"  data samples: {N}")
    print(f"  Laguerre pole a: {args.a}, M: {args.M}")
    print(f"  Volterra order: {args.order}")
    print(f"  Design matrix columns: {X.shape[1]}")
    print(f"  Ridge alpha: {args.alpha}")
    print("  Training metrics:")
    print(f"    RMSE: {m['RMSE']:.6g}")
    print(f"    NMSE: {m['NMSE']:.6g}")
    print(f"    R^2 : {m['R2']:.6g}")

    # indicate kernel sizes
    print(
        "  Kernel reconstructions (truncated impulse length = {}):".format(
            args.impulse_length
        )
    )
    print(f"    h1: {h1.shape if h1 is not None else None}")
    if args.order >= 2:
        print(f"    h2: {h2.shape if h2 is not None else None}")
    if args.order >= 3:
        print(f"    h3: {h3.shape if h3 is not None else None} (may be large)")

    # save model if desired
    if not args.no_save:
        fn = args.save
        # ensure extension .npz
        if not fn.lower().endswith(".npz"):
            fn = fn + ".npz"
        save_model(fn, model, args.a, args.M, args.order, info)
        print(f"  Model saved to: {os.path.abspath(fn)}")

    # plot comparison if desired
    if not args.no_plot_compare:
        plt.figure(figsize=(10, 5))
        plt.plot(y, label="Output (y)", linewidth=1)
        plt.plot(yhat, label="Prediction (y_hat)", linewidth=1, linestyle="--")
        plt.title("Output vs Prediction")
        plt.xlabel("Sample index")
        plt.ylabel("Output")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
