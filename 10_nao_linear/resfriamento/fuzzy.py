from __future__ import annotations
import argparse
import numpy as np
import os
from skfuzzy.cluster import cmeans
import matplotlib.pyplot as plt
from functools import partial
import itertools

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
                  centers: np.ndarray, U_mem: np.ndarray, m_exp: float = 1.0,
                  intercept: bool = True, ridge: float = 0.0,
                  model_delta: bool = False, stability_rho: float | None = None) -> dict:
    """
    Extended: ridge (>=0), model_delta -> regress Δx = x_{k+1}-x_k, stability_rho caps spectral radius of A.
    """
    nc, N = U_mem.shape
    n_x = X.shape[1]
    if Y.shape[0] != N or U_in.shape[0] != N:
        raise ValueError("Shapes mismatch among X, U_in, Y and membership matrix.")
    # Target: either Y or Δx
    if model_delta:
        Y_target = Y - X
    else:
        Y_target = Y
    cols = [X, U_in.reshape(-1, 1)]
    if intercept:
        cols.append(np.ones((N, 1)))
    Phi = np.concatenate(cols, axis=1)  # (N, d)
    d = Phi.shape[1]
    A_all = np.zeros((nc, n_x, n_x))
    B_all = np.zeros((nc, n_x))
    c_all = np.zeros((nc, n_x))
    Theta_all = np.zeros((nc, d, n_x))
    Ireg = ridge * np.eye(d)
    for r in range(nc):
        weights = (U_mem[r, :] ** m_exp)
        Wsqrt = np.sqrt(weights)
        Phi_w = Phi * Wsqrt[:, None]
        Y_w = Y_target * Wsqrt[:, None]
        # Ridge solution: (Phi_w^T Phi_w + λI) Θ = Phi_w^T Y_w
        M = Phi_w.T @ Phi_w + Ireg
        RHS = Phi_w.T @ Y_w
        Theta_r = np.linalg.solve(M, RHS)  # (d, n_x)
        Theta_all[r, :, :] = Theta_r
        Theta_X = Theta_r[0:n_x, :]
        Theta_u = Theta_r[n_x:n_x+1, :]
        A_r = Theta_X.T
        B_r = Theta_u.flatten().T
        Theta_c = Theta_r[-1, :] if intercept else np.zeros((n_x,))
        # Stability projection (only meaningful if not delta model or even with delta for Jacobian)
        if stability_rho is not None and stability_rho > 0:
            eigvals, eigvecs = np.linalg.eig(A_r)
            rho = max(abs(eigvals))
            if rho > stability_rho:
                A_r = A_r * (stability_rho / rho)
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
        'intercept': intercept,
        'ridge': ridge,
        'model_delta': model_delta,
        'stability_rho': stability_rho
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
    model_delta = bool(model.get('model_delta', False))
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
        if model_delta:
            delta = A_blend.dot(x) + B_blend * u_k + c_blend
            x = x + delta
        else:
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

# Helpers to parse lists from CLI (comma-separated)
def _parse_list(s: str | None, cast):
    if s is None:
        return None
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok == '':
            continue
        vals.append(cast(tok))
    return vals

def _parse_bool_list(s: str | None):
    if s is None:
        return None
    true_set = {'1','true','yes','y','t'}
    false_set = {'0','false','no','n','f'}
    vals = []
    for tok in s.split(','):
        t = tok.strip().lower()
        if t in true_set:
            vals.append(True)
        elif t in false_set:
            vals.append(False)
        else:
            raise ValueError(f"Invalid boolean token in list: {tok}")
    return vals

def evaluate_training(X: np.ndarray, U_in: np.ndarray, Y: np.ndarray, model: dict, premise_func) -> dict:
    """
    Adjusted for delta modeling: if model_delta True, free-run works transparently.
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
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Fraction (0..1) of one-step pairs reserved for test. Default 0.2.')
    parser.add_argument('--no-plot-test', action='store_true', help='Do not plot test comparison.')
    # Hyperparameter optimization controls
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter grid search with validation split.')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Fraction of training pairs reserved for validation when optimizing. Default 0.2.')
    parser.add_argument('--hp-nc', type=str, default=None, help='Comma-separated candidates for number of clusters, e.g., "2,3,4".')
    parser.add_argument('--hp-m', type=str, default=None, help='Comma-separated candidates for fuzzifier m, e.g., "1.8,2.0,2.2".')
    parser.add_argument('--hp-weight-exp', type=str, default=None, help='Comma-separated candidates for weight exponent, e.g., "1.0,2.0".')
    parser.add_argument('--hp-intercept', type=str, default=None, help='Comma-separated booleans for intercept, e.g., "true,false".')
    parser.add_argument('--hp-include-input-premise', type=str, default=None, help='Comma-separated booleans for include-input-premise, e.g., "true,false".')
    parser.add_argument('--standardize', action='store_true', help='Standardize states (and input) before clustering/regression.')
    parser.add_argument('--model-delta', action='store_true', help='Model Δx instead of absolute next x.')
    parser.add_argument('--ridge', type=float, default=0.0, help='Ridge regularization λ.')
    parser.add_argument('--stability-rho', type=float, default=None, help='Cap spectral radius of each local A (e.g., 0.995).')
    args = parser.parse_args()

    u, states = read_data(args.file)
    # Standardization stats
    if args.standardize:
        x_mean = states.mean(axis=0)
        x_std = states.std(axis=0)
        x_std[x_std == 0] = 1.0
        u_mean = u.mean()
        u_std = u.std()
        if u_std == 0: u_std = 1.0
        states_scaled = (states - x_mean) / x_std
        u_scaled = (u - u_mean) / u_std
    else:
        states_scaled = states
        u_scaled = u
        x_mean = x_std = u_mean = u_std = None
    X, U_in, Y = prepare_sequence(states_scaled, u_scaled)    # X: (N-1,5), U_in: (N-1,), Y: (N-1,5)
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

    best_cfg = None
    model = None
    premise_func = None
    used_include_input_premise = args.include_input_premise
    used_nc = args.nc
    used_m = args.m
    used_weight_exp = args.weight_exp
    used_intercept = args.intercept

    if args.optimize:
        # Build candidate lists (use provided or small defaults around typical values)
        nc_list = _parse_list(args.hp_nc, int) if args.hp_nc is not None else sorted(set([max(2, args.nc-1), args.nc, args.nc+1]))
        m_list = _parse_list(args.hp_m, float) if args.hp_m is not None else sorted(set([round(v,3) for v in [args.m, 1.8, 2.0, 2.2]]))
        wexp_list = _parse_list(args.hp_weight_exp, float) if args.hp_weight_exp is not None else sorted(set([args.weight_exp, 1.0, args.m]))
        intercept_list = _parse_bool_list(args.hp_intercept) if args.hp_intercept is not None else [False, True]
        include_premise_list = _parse_bool_list(args.hp_include_input_premise) if args.hp_include_input_premise is not None else [False, True]

        # Avoid exploding grid sizes
        comb_count = (len(nc_list or []) *
                      len(m_list or []) *
                      len(wexp_list or []) *
                      len(intercept_list or []) *
                      len(include_premise_list or []))
        if comb_count > 200:
            include_premise_list = [args.include_input_premise]
            intercept_list = [args.intercept]
            comb_count = len(nc_list or [])*len(m_list or [])*len(wexp_list or [])*len(intercept_list or [])*len(include_premise_list or [])

        # Validation split from training (chronological)
        if not (0.0 < args.val_ratio < 1.0):
            raise ValueError("val-ratio must be in (0, 1) when optimizing.")
        val_len = max(1, int(np.floor(N_tr * args.val_ratio)))
        if N_tr - val_len <= 0:
            raise ValueError("Not enough training samples to create validation split; reduce --val-ratio or provide more data.")
        X_fit, U_fit, Y_fit = X_tr[:N_tr-val_len], U_tr[:N_tr-val_len], Y_tr[:N_tr-val_len]
        X_val, U_val, Y_val = X_tr[N_tr-val_len:], U_tr[N_tr-val_len:], Y_tr[N_tr-val_len:]

        best_rmse = np.inf
        # Ensure all lists are not None
        nc_list = nc_list if nc_list is not None else []
        m_list = m_list if m_list is not None else []
        wexp_list = wexp_list if wexp_list is not None else []
        intercept_list = intercept_list if intercept_list is not None else []
        include_premise_list = include_premise_list if include_premise_list is not None else []

        # Grid search
        for nc_v, m_v, wexp_v, intercept_v, incprem_v in itertools.product(nc_list, m_list, wexp_list, intercept_list, include_premise_list):
            # Prepare premise on fit subset
            if incprem_v:
                premise_fit = np.concatenate([X_fit, U_fit.reshape(-1,1)], axis=1)
            else:
                premise_fit = X_fit
            premise_fit_T = premise_fit.T

            try:
                cntr, U_mem_fit = fuzzy_cmeans(premise_fit_T, nc=nc_v, m=m_v)
                mdl = fit_ts_models(X_fit, U_fit, Y_fit, centers=cntr, U_mem=U_mem_fit, m_exp=wexp_v, intercept=intercept_v)
                pf = partial(premise_membership_from_cmeans, fuzzifier=m_v)
                eval_val = evaluate_training(X_val, U_val, Y_val, mdl, pf)
                rmse_val = float(eval_val['overall_rmse'])
            except Exception as e:
                # Skip invalid configurations
                continue

            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_cfg = {
                    'nc': nc_v,
                    'm': m_v,
                    'weight_exp': wexp_v,
                    'intercept': intercept_v,
                    'include_input_premise': incprem_v,
                    'val_overall_rmse': best_rmse
                }

        if best_cfg is None:
            raise RuntimeError("Hyperparameter optimization failed to find a valid configuration.")

        # Refit best model on full training subset
        used_nc = best_cfg['nc']
        used_m = best_cfg['m']
        used_weight_exp = best_cfg['weight_exp']
        used_intercept = best_cfg['intercept']
        used_include_input_premise = best_cfg['include_input_premise']

        if used_include_input_premise:
            premise_tr = np.concatenate([X_tr, U_tr.reshape(-1,1)], axis=1)
        else:
            premise_tr = X_tr
        premise_tr_T = premise_tr.T
        cntr, U_mem_tr = fuzzy_cmeans(premise_tr_T, nc=used_nc, m=used_m)
        model = fit_ts_models(X_tr, U_tr, Y_tr, centers=cntr, U_mem=U_mem_tr, m_exp=used_weight_exp, intercept=used_intercept)
        premise_func = partial(premise_membership_from_cmeans, fuzzifier=used_m)

        # Evaluate train/test
        eval_tr = evaluate_training(X_tr, U_tr, Y_tr, model, premise_func)
        rmse_per_state_tr = eval_tr['rmse_per_state']
        overall_rmse_tr = eval_tr['overall_rmse']

        eval_te = None
        if N_te > 0:
            eval_te = evaluate_training(X_te, U_te, Y_te, model, premise_func)
            rmse_per_state_te = eval_te['rmse_per_state']
            overall_rmse_te = eval_te['overall_rmse']

        # Report
        print("Takagi-Sugeno state-space estimation report")
        print(f"  total one-step pairs: {N}  |  train: {N_tr}  test: {N_te}")
        print("  Hyperparameter optimization: ENABLED")
        print(f"  best config -> nc: {used_nc}, m: {used_m}, weight_exp: {used_weight_exp}, intercept: {used_intercept}, include_input_premise: {used_include_input_premise}")
        print(f"  validation overall RMSE (free-run): {best_cfg['val_overall_rmse']:.6g}")
        print("  Training RMSE per state (t0..t4):")
        for i, v in enumerate(rmse_per_state_tr):
            print(f"    t{i}: {v:.6g}")
        print(f"  Training overall RMSE: {overall_rmse_tr:.6g}")
        if eval_te is not None:
            print("  Test RMSE per state (t0..t4):")
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
                'nc': used_nc,
                'm': used_m,
                'weight_exp': used_weight_exp,
                'include_input_premise': used_include_input_premise,
                'intercept': used_intercept,
                'train_samples': int(N_tr),
                'test_samples': int(N_te),
                'test_ratio': float(args.test_ratio),
                'optimize': True,
                'val_ratio': float(args.val_ratio),
                'val_overall_rmse': float(best_cfg['val_overall_rmse']),
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
            plt.text(0.5, 0.01, f'Config: nc={used_nc}, m={used_m}, weight_exp={used_weight_exp}, intercept={used_intercept}, include_input_premise={used_include_input_premise}',
                     horizontalalignment='center', verticalalignment='bottom', transform=plt.gcf().transFigure)
            plt.suptitle('Treino: Real vs Previsão do Modelo TS')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.savefig('comparacao_treino.png')
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
                plt.text(0.5, 0.01, f'Config: nc={used_nc}, m={used_m}, weight_exp={used_weight_exp}, intercept={used_intercept}, include_input_premise={used_include_input_premise}',
                         horizontalalignment='center', verticalalignment='bottom', transform=plt.gcf().transFigure)
                plt.suptitle('Teste: Real vs Previsão do Modelo TS')
                plt.tight_layout(rect=(0, 0.03, 1, 0.95))
                plt.savefig('comparacao_teste.png')
                plt.show()
        return

    # ========== Baseline path ==========
    # Build premise data for clustering: use only training subset
    if used_include_input_premise:
        premise_tr = np.concatenate([X_tr, U_tr.reshape(-1,1)], axis=1)  # (N_tr, feat)
    else:
        premise_tr = X_tr  # (N_tr, feat)
    premise_tr_T = premise_tr.T  # shape (feat, N_tr) required by skfuzzy

    # clustering on training
    cntr, U_mem_tr = fuzzy_cmeans(premise_tr_T, nc=used_nc, m=used_m)

    # Fit model on training data
    model = fit_ts_models(X_tr, U_tr, Y_tr, centers=cntr, U_mem=U_mem_tr, m_exp=used_weight_exp, intercept=used_intercept)

    # choose premise membership function for simulation/evaluation (bind fuzzifier m)
    premise_func = partial(premise_membership_from_cmeans, fuzzifier=used_m)

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
    print("  Hyperparameter optimization: DISABLED")
    print(f"  clusters (rules): {used_nc}, fuzzifier m: {used_m}, weight exponent: {used_weight_exp}")
    print(f"  include input in premise: {used_include_input_premise}, intercept: {used_intercept}")
    print("  Training RMSE per state (t0..t4):")
    for i, v in enumerate(rmse_per_state_tr):
        print(f"    t{i}: {v:.6g}")
    print(f"  Training overall RMSE: {overall_rmse_tr:.6g}")
    if eval_te is not None:
        print("  Test RMSE per state (t0..t4):")
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
            'nc': used_nc,
            'm': used_m,
            'weight_exp': used_weight_exp,
            'include_input_premise': used_include_input_premise,
            'intercept': used_intercept,
            'train_samples': int(N_tr),
            'test_samples': int(N_te),
            'test_ratio': float(args.test_ratio),
            'optimize': False,
        }
        meta_extra = {
            'standardize': args.standardize,
            'model_delta': args.model_delta,
            'ridge': args.ridge,
            'stability_rho': args.stability_rho
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
        plt.savefig('comparacao_treino.png')
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
            plt.savefig('comparacao_teste.png')
            plt.show()

if __name__ == "__main__":
    main()
