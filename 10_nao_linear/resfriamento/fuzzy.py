from __future__ import annotations
import argparse
import numpy as np
import os
from skfuzzy.cluster import cmeans
import matplotlib.pyplot as plt
from functools import partial

def read_data(fname: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(fname)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    u = d[:, 0].astype(float)
    states = d[:, 1:6].astype(float)
    if states.shape[1] != 5:
        raise ValueError("Expected five temperature columns (t0..t4) after input column.")
    return u, states

def prepare_sequence(states: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build regression sequences X = x_k, U = u_k, Y = x_{k+1}
    Drops last sample for X,U and first sample for Y.
    """
    X = states[:-1, :]    # shape (N-1, n_states)
    U = u[:-1]            # shape (N-1,)
    Y = states[1:, :]     # shape (N-1, n_states)
    return X, U, Y

def fuzzy_cmeans(premise: np.ndarray, nc: int, m: float = 2.0, error: float = 1e-5, maxiter: int = 1000):
    """
    Run fuzzy c-means on premise data (shape: features x samples).
    Returns cluster centers (nc x features) and membership matrix U (nc x samples).
    """
    cntr, U, _, _, _, _, _ = cmeans(
        premise, c=nc, m=m, error=error, maxiter=maxiter, init=None
    )
    # cntr: (nc, features), U: (nc, samples)
    return cntr, U

def fit_ts_models(X: np.ndarray, U_in: np.ndarray, Y: np.ndarray,
                  centers: np.ndarray, U_mem: np.ndarray, m_exp: float = 1.0, intercept: bool = True) -> dict:
    """
    Fit TS models:
      - X: (N, n_x)
      - U_in: (N,) input
      - Y: (N, n_x)
      - centers: (nc, n_premise_features)
      - U_mem: (nc, N) membership matrix from clustering (rows sum to 1)
      - m_exp: exponent to raise membership to use as weights (commonly = 1 or = fuzzifier m)
    Returns dictionary with:
      - 'A': (nc, n_x, n_x)
      - 'B': (nc, n_x)   (single-input assumed)
      - 'c': (nc, n_x)
      - 'Theta': (nc, d, n_x) where d = n_x + 1 (+1 for intercept if used)
      - 'centers', 'U_mem', 'm_exp'
    """
    nc, N = U_mem.shape
    n_x = X.shape[1]
    if Y.shape[0] != N or U_in.shape[0] != N:
        raise ValueError("Shapes mismatch among X, U_in, Y and membership matrix.")
    # Build regressor Phi = [X, u, (1)] -> shape (N, d)
    cols = [X, U_in.reshape(-1, 1)]
    if intercept:
        cols.append(np.ones((N, 1)))
    Phi = np.concatenate(cols, axis=1)  # (N, d)
    d = Phi.shape[1]
    A_all = np.zeros((nc, n_x, n_x))
    B_all = np.zeros((nc, n_x))
    c_all = np.zeros((nc, n_x))
    Theta_all = np.zeros((nc, d, n_x))
    # For each cluster r solve weighted LS: minimize sum_n w_{r,n} ||Y_n - Phi_n Theta_r||^2
    for r in range(nc):
        weights = (U_mem[r, :] ** m_exp)  # length N
        # Weighted least squares via multiplication by sqrt(weights)
        Wsqrt = np.sqrt(weights)
        Phi_w = Phi * Wsqrt[:, None]     # (N, d)
        Y_w = Y * Wsqrt[:, None]         # (N, n_x)
        # Solve Theta via lstsq: Phi_w @ Theta = Y_w
        Theta_r, *_ = np.linalg.lstsq(Phi_w, Y_w, rcond=None)
        # Theta_r shape (d, n_x)
        Theta_all[r, :, :] = Theta_r
        # extract A, B, c
        Theta_X = Theta_r[0:n_x, :]     # (n_x, n_x)
        Theta_u = Theta_r[n_x:n_x+1, :] # (1, n_x)
        A_r = Theta_X.T                 # (n_x, n_x) so that x_next = A_r @ x + B_r * u + c_r
        B_r = Theta_u.flatten().T       # (n_x,)
        if intercept:
            Theta_c = Theta_r[-1, :]    # (n_x,)
        else:
            Theta_c = np.zeros((n_x,))
        A_all[r, :, :] = A_r
        B_all[r, :] = B_r
        c_all[r, :] = Theta_c
    return {
        'A': A_all,
        'B': B_all,
        'c': c_all,
        'Theta': Theta_all,
        'centers': centers,
        'U_mem': U_mem,
        'm_exp': m_exp,
        'intercept': intercept
    }

def predict_ts_sequence(x0: np.ndarray, u_seq: np.ndarray, model: dict, premise_func, return_all_states: bool = True) -> np.ndarray:
    """
    Simulate TS model open-loop for a sequence of inputs starting from x0.
    - x0: (n_x,) initial state
    - u_seq: (T,) sequence of inputs; produces T+1 states (including x0) if you want.
    - model contains A,B,c and centers used by premise_func to compute memberships for given (x,u).
    - premise_func(point, centers) -> U_new (nc,) membership row for point
    Returns states array of shape (T+1, n_x).
    """
    A_all = model['A']
    B_all = model['B']
    c_all = model['c']
    centers = model['centers']
    nc = A_all.shape[0]
    n_x = x0.shape[0]
    T = u_seq.shape[0]
    Xpred = np.zeros((T + 1, n_x))
    x = x0.copy()
    Xpred[0, :] = x
    for k in range(T):
        u_k = float(u_seq[k])
        # compute membership for current premise point (x,u)
        U_row = premise_func(x, u_k, centers)  # expected shape (nc,)
        # normalize (numerical safety)
        U_row = np.maximum(U_row, 1e-12)
        U_row = U_row / np.sum(U_row)
        # blend local models
        A_blend = np.tensordot(U_row, A_all, axes=(0,0))  # (n_x, n_x)
        B_blend = np.tensordot(U_row, B_all, axes=(0,0))  # (n_x,)
        c_blend = np.tensordot(U_row, c_all, axes=(0,0))  # (n_x,)
        # propagate
        x = A_blend.dot(x) + B_blend * u_k + c_blend
        Xpred[k+1, :] = x
    return Xpred

def premise_membership_from_cmeans(x: np.ndarray, u_k: float, centers: np.ndarray, fuzzifier: float = 2.0):
    """
    Compute membership row for a single point using fuzzy c-means distance formula:
    centers shape: (nc, feat)
    premise vector is constructed same as used for clustering: e.g., [x, u] or only x.
    Here we assume clustering included u in the same ordering (so centers columns correspond).
    The fuzzy c-means membership formula (used when computing U for unseen data):
      u_i = 1 / sum_j (||x-c_i||/||x-c_j||)^{2/(m-1)}
    """
    # build vector p
    p = np.concatenate([x, np.array([u_k])]) if centers.shape[1] == x.size + 1 else x
    # distances
    eps = 1e-12
    d = np.linalg.norm(centers - p[None, :], axis=1)
    d = np.maximum(d, eps)
    power = 2.0 / (fuzzifier - 1.0) if fuzzifier > 1.0 else 2.0
    inv = (1.0 / d) ** power
    U_row = inv / np.sum(inv)
    return U_row

def premise_membership_from_kmeans(x: np.ndarray, u_k: float, centers: np.ndarray, fuzzifier: float = 2.0):
    # same formula works for centers from KMeans fallback
    return premise_membership_from_cmeans(x, u_k, centers, fuzzifier=fuzzifier)

def evaluate_training(X: np.ndarray, U_in: np.ndarray, Y: np.ndarray, model: dict, premise_func) -> dict:
    """
    Evaluate model on training data using open-loop (free-run) simulation.
    Starts from x0 = X[0] and simulates forward with U_in; compares predictions 1..N to Y.
    Returns RMSE per state and overall.
    """
    N = X.shape[0]
    if N == 0:
        raise ValueError("Empty training set.")
    x0 = X[0, :]
    # simulate free-run with measured inputs
    Xpred = predict_ts_sequence(x0, U_in, model, premise_func, return_all_states=True)  # shape (N+1, n_x)
    Yhat = Xpred[1:, :]  # align with measured Y (states[1:,:])
    err = Y - Yhat
    rmse_per_state = np.sqrt(np.mean(err**2, axis=0))
    overall_rmse = np.sqrt(np.mean(err**2))
    return {'Yhat': Yhat, 'Xpred': Xpred, 'rmse_per_state': rmse_per_state, 'overall_rmse': overall_rmse}

def save_model_npz(fn: str, model: dict, meta: dict):
    os.makedirs(os.path.dirname(fn), exist_ok=True) if os.path.dirname(fn) else None
    np.savez(fn,
             A = model['A'],
             B = model['B'],
             c = model['c'],
             Theta = model['Theta'],
             centers = model['centers'],
             U_mem = model['U_mem'],
             **{f'meta_{k}': v for k, v in meta.items()})

def main():
    parser = argparse.ArgumentParser(description="Estimate TS state-space model for t0..t4 states.")
    parser.add_argument('file', help='Input file (.txt): u t0 t1 t2 t3 t4 ...')
    parser.add_argument('--nc', type=int, default=3, help='Number of fuzzy clusters (rules). Default 3.')
    parser.add_argument('--m', type=float, default=2.0, help='Fuzzifier m for fuzzy-cmeans. Default 2.0.')
    parser.add_argument('--weight-exp', type=float, default=1.0, help='Exponent applied to membership when weighting LS. Default 1.')
    parser.add_argument('--include-input-premise', action='store_true', help='Use input u together with states as premise variables.')
    parser.add_argument('--intercept', action='store_true', help='Include intercept term in local linear models.')
    parser.add_argument('--save', type=str, default='ts_model.npz', help='Filename to save estimated model (.npz).')
    parser.add_argument('--no-save', action='store_true', help='Do not save model file.')
    parser.add_argument('--no-plot-compare', action='store_true', help='Plot training data vs model predictions.')
    # NEW: split/plot-test controls
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Fraction (0..1) of one-step pairs reserved for test. Default 0.2.')
    parser.add_argument('--no-plot-test', action='store_true', help='Do not plot test comparison.')
    args = parser.parse_args()

    u, states = read_data(args.file)
    X, U_in, Y = prepare_sequence(states, u)    # X: (N-1,5), U_in: (N-1,), Y: (N-1,5)
    N = X.shape[0]
    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("test-ratio must be in [0, 1).")
    test_len = int(np.floor(N * args.test_ratio))
    split_idx = N - test_len
    if split_idx <= 0:
        raise ValueError("Train set would be empty; reduce --test-ratio.")

    # Split into train/test (chronological)
    X_tr, U_tr, Y_tr = X[:split_idx], U_in[:split_idx], Y[:split_idx]
    X_te, U_te, Y_te = X[split_idx:], U_in[split_idx:], Y[split_idx:]
    N_tr, N_te = X_tr.shape[0], X_te.shape[0]

    # Build premise data for clustering: use only training subset
    if args.include_input_premise:
        premise_tr = np.concatenate([X_tr, U_tr.reshape(-1,1)], axis=1)  # (N_tr, feat)
    else:
        premise_tr = X_tr  # (N_tr, feat)
    premise_tr_T = premise_tr.T  # shape (feat, N_tr) required by skfuzzy

    # clustering on training
    cntr, U_mem_tr = fuzzy_cmeans(premise_tr_T, nc=args.nc, m=args.m)

    # Fit model on training data
    model = fit_ts_models(X_tr, U_tr, Y_tr, centers=cntr, U_mem=U_mem_tr, m_exp=args.weight_exp, intercept=args.intercept)

    # choose premise membership function for simulation/evaluation (bind fuzzifier m)
    premise_func = partial(premise_membership_from_cmeans, fuzzifier=args.m)

    # training evaluation (free-run/open-loop)
    eval_tr = evaluate_training(X_tr, U_tr, Y_tr, model, premise_func)
    rmse_per_state_tr = eval_tr['rmse_per_state']
    overall_rmse_tr = eval_tr['overall_rmse']

    # test evaluation if available
    eval_te = None
    if N_te > 0:
        eval_te = evaluate_training(X_te, U_te, Y_te, model, premise_func)
        rmse_per_state_te = eval_te['rmse_per_state']
        overall_rmse_te = eval_te['overall_rmse']

    # print concise report
    print("Takagi-Sugeno state-space estimation report")
    print(f"  total one-step pairs: {N}  |  train: {N_tr}  test: {N_te}")
    print(f"  clusters (rules): {args.nc}, fuzzifier m: {args.m}, weight exponent: {args.weight_exp}")
    print(f"  include input in premise: {args.include_input_premise}, intercept: {args.intercept}")
    print("  Training free-run RMSE per state (t0..t4):")
    for i, v in enumerate(rmse_per_state_tr):
        print(f"    t{i}: {v:.6g}")
    print(f"  Training overall RMSE: {overall_rmse_tr:.6g}")
    if eval_te is not None:
        print("  Test free-run RMSE per state (t0..t4):")
        for i, v in enumerate(rmse_per_state_te):
            print(f"    t{i}: {v:.6g}")
        print(f"  Test overall RMSE: {overall_rmse_te:.6g}")

    # Save model
    if not args.no_save:
        fn = args.save
        if not fn.lower().endswith('.npz'):
            fn = fn + '.npz'
        meta = {
            'file': args.file,
            'nc': args.nc,
            'm': args.m,
            'weight_exp': args.weight_exp,
            'include_input_premise': args.include_input_premise,
            'intercept': args.intercept,
            'train_samples': int(N_tr),
            'test_samples': int(N_te),
            'test_ratio': float(args.test_ratio),
        }
        save_model_npz(fn, model, meta)
        print(f"  Model saved to: {os.path.abspath(fn)}")

    # Plot comparison
    if not args.no_plot_compare:
        # Train plot
        Yhat_tr = eval_tr['Yhat']
        time_tr = np.arange(N_tr)
        plt.figure(figsize=(12, 8))
        for i in range(states.shape[1]):
            plt.subplot(states.shape[1], 1, i+1)
            plt.plot(time_tr, Y_tr[:, i], label=f't{i} real', color='blue')
            plt.plot(time_tr, Yhat_tr[:, i], label=f't{i} estimado', color='red', linestyle='--')
            plt.ylabel(f'T{i}')
            plt.legend()
            plt.grid()
        plt.xlabel('Amostra')
        plt.suptitle('Treino: Real vs Previsão do Modelo TS')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.show()

        # Test plot
        if (eval_te is not None) and (not args.no_plot_test):
            Yhat_te = eval_te['Yhat']
            time_te = np.arange(N_te)
            plt.figure(figsize=(12, 8))
            for i in range(states.shape[1]):
                plt.subplot(states.shape[1], 1, i+1)
                plt.plot(time_te, Y_te[:, i], label=f't{i} real', color='blue')
                plt.plot(time_te, Yhat_te[:, i], label=f't{i} estimado', color='red', linestyle='--')
                plt.ylabel(f'T{i}')
                plt.legend()
                plt.grid()
            plt.xlabel('Amostra')
            plt.suptitle('Teste: Real vs Previsão do Modelo TS')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()

if __name__ == "__main__":
    main()
