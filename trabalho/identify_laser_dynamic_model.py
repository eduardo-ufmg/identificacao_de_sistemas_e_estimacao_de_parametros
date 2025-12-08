import argparse
import json
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures


def build_laser_features(scan: np.ndarray) -> np.ndarray:
    """Extract simple summary features from one laser scan."""
    n = scan.size
    if n == 0:
        return np.zeros(6, dtype=float)
    third = max(1, n // 3)
    left = scan[:third]
    front = scan[third : 2 * third]
    right = scan[2 * third :]
    return np.array(
        [
            scan.mean(),
            scan.min(),
            scan.std(),
            left.mean(),
            front.mean() if front.size else scan.mean(),
            right.mean() if right.size else scan.mean(),
        ],
        dtype=float,
    )


def load_series(ref_path: str, laser_path: str):
    ref = np.loadtxt(ref_path, delimiter=",")
    laser = np.loadtxt(laser_path, delimiter=",")
    if ref.ndim != 2 or ref.shape[1] < 3:
        raise ValueError("Reference data must have at least 3 columns (x, y, theta)")
    if laser.ndim != 2:
        raise ValueError("Laser data must be 2-D (time x beams)")
    length = min(len(ref), len(laser))
    if length < 2:
        raise ValueError("Not enough samples to fit a dynamic model")
    return ref[:length], laser[:length]


def fit_state_space(ref: np.ndarray, laser: np.ndarray, step: int):
    states = ref[:, :3]
    states_next = states[step:]
    states_now = states[:-step]
    laser_now = laser[:-step]

    features = np.apply_along_axis(build_laser_features, 1, laser_now)
    X = np.hstack((states_now, features))
    y = states_next

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    coef = model.coef_  # shape (3, features)
    intercept = model.intercept_

    n_state = states_now.shape[1]
    A = coef[:, :n_state]
    B = coef[:, n_state:]

    pred = model.predict(X)
    rmse = np.sqrt(np.mean((pred - y) ** 2, axis=0))

    return A, B, intercept, rmse, model


def fit_narx_model(
    ref: np.ndarray,
    laser: np.ndarray,
    step: int,
    n_lags: int = 3,
    model_type: str = "linear",
    poly_degree: int = 2,
    hidden_layers: tuple = (50, 50),
    alpha: float = 0.01,
):
    """
    Fit a NARX (Nonlinear AutoRegressive with eXogenous inputs) model.

    Args:
        ref: Reference trajectory (n_samples, 3) with [x, y, theta]
        laser: Laser scans (n_samples, n_beams)
        step: Time step delta for the model
        n_lags: Number of past time steps to use (autoregressive order)
        model_type: 'linear', 'polynomial', or 'neural'
        poly_degree: Polynomial degree for nonlinear features (if model_type='polynomial')
        hidden_layers: Hidden layer sizes for neural network (if model_type='neural')
        alpha: Regularization parameter

    Returns:
        model: Fitted NARX model object
        poly_features: PolynomialFeatures object (if applicable) or None
        rmse: RMSE per state component
        config: Dictionary with NARX configuration
    """
    states = ref[:, :3]

    # Build NARX input features: past states and laser features
    n_samples = len(states) - n_lags - step + 1

    if n_samples < 10:
        raise ValueError(
            f"Not enough samples for n_lags={n_lags}, step={step}. Need at least {n_lags + step + 10} samples."
        )

    # Construct lagged inputs
    X_list = []
    y_list = []

    for i in range(n_samples):
        # Past states: from i to i+n_lags-1
        past_states = states[i : i + n_lags].flatten()  # Shape: (n_lags * 3,)

        # Current laser features at time i+n_lags-1
        current_laser = laser[i + n_lags - 1]
        laser_features = build_laser_features(current_laser)  # Shape: (6,)

        # Concatenate
        x_i = np.concatenate([past_states, laser_features])
        X_list.append(x_i)

        # Target: state at time i+n_lags-1+step
        y_i = states[i + n_lags - 1 + step]
        y_list.append(y_i)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"NARX data shape: X={X.shape}, y={y.shape}")

    # Fit model based on type
    poly_features = None

    if model_type == "linear":
        # Linear NARX with Ridge regularization
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y)

    elif model_type == "polynomial":
        # Polynomial NARX
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        print(
            f"Polynomial features: {X_poly.shape[1]} features from degree-{poly_degree} expansion"
        )

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_poly, y)

    elif model_type == "neural":
        # Neural Network NARX
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            max_iter=1000,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
        model.fit(X, y)
        print(f"Neural network trained: {len(model.loss_curve_)} iterations")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute predictions and RMSE
    if model_type == "polynomial" and poly_features is not None:
        X_pred = poly_features.transform(X)
        pred = model.predict(X_pred)
    else:
        pred = model.predict(X)

    rmse = np.sqrt(np.mean((y - pred) ** 2, axis=0))

    # Configuration dict
    config = {
        "n_lags": n_lags,
        "step": step,
        "model_type": model_type,
        "poly_degree": poly_degree if model_type == "polynomial" else None,
        "hidden_layers": list(hidden_layers) if model_type == "neural" else None,
        "alpha": alpha,
        "n_features": X.shape[1],
    }

    return model, poly_features, rmse, config


def grid_search_narx(
    ref: np.ndarray,
    laser: np.ndarray,
    step: int,
    param_grid: dict,
    n_splits: int = 5,
    verbose: bool = True,
):
    """
    Perform grid search over NARX hyperparameters with time series cross-validation.

    Args:
        ref: Reference trajectory (n_samples, 3)
        laser: Laser scans (n_samples, n_beams)
        step: Time step delta
        param_grid: Dictionary with hyperparameter lists:
            - 'n_lags': list of lag values
            - 'model_type': list of model types
            - 'poly_degree': list of polynomial degrees (for polynomial models)
            - 'hidden_layers': list of hidden layer tuples (for neural models)
            - 'alpha': list of regularization values
        n_splits: Number of time series cross-validation splits
        verbose: Print progress

    Returns:
        best_params: Dictionary with best hyperparameters
        best_score: Best average RMSE score
        results: List of all results with params and scores
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_results = []
    best_score = float("inf")
    best_params = None

    total_combinations = np.prod([len(v) for v in param_values])
    print(f"\nGrid Search: Testing {total_combinations} parameter combinations")
    print(f"Cross-validation: {n_splits} splits")
    print("=" * 80)

    for i, param_combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, param_combo))

        # Skip invalid combinations
        if params["model_type"] == "polynomial" and "poly_degree" not in params:
            continue
        if params["model_type"] == "neural" and "hidden_layers" not in params:
            continue

        if verbose:
            print(f"\n[{i}/{total_combinations}] Testing: {params}")

        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []

            states = ref[:, :3]
            min_samples_needed = params["n_lags"] + step + 10

            for fold, (train_idx, val_idx) in enumerate(tscv.split(states)):
                # Ensure we have enough samples
                if (
                    len(train_idx) < min_samples_needed
                    or len(val_idx) < min_samples_needed
                ):
                    continue

                train_ref = ref[train_idx]
                train_laser = laser[train_idx]
                val_ref = ref[val_idx]
                val_laser = laser[val_idx]

                # Fit model on training data
                model, poly_features, _, config = fit_narx_model(
                    train_ref,
                    train_laser,
                    step,
                    n_lags=params["n_lags"],
                    model_type=params["model_type"],
                    poly_degree=params.get("poly_degree", 2),
                    hidden_layers=params.get("hidden_layers", (50, 50)),
                    alpha=params["alpha"],
                )

                # Evaluate on validation data
                val_states = val_ref[:, :3]
                n_val_samples = len(val_states) - params["n_lags"] - step + 1

                if n_val_samples < 1:
                    continue

                X_val_list = []
                y_val_list = []

                for j in range(n_val_samples):
                    past_states = val_states[j : j + params["n_lags"]].flatten()
                    current_laser = val_laser[j + params["n_lags"] - 1]
                    laser_features = build_laser_features(current_laser)
                    x_val = np.concatenate([past_states, laser_features])
                    X_val_list.append(x_val)
                    y_val_list.append(val_states[j + params["n_lags"] - 1 + step])

                X_val = np.array(X_val_list)
                y_val = np.array(y_val_list)

                # Predict
                if params["model_type"] == "polynomial" and poly_features is not None:
                    X_val_transformed = poly_features.transform(X_val)
                    val_pred = model.predict(X_val_transformed)
                else:
                    val_pred = model.predict(X_val)

                # Compute RMSE
                fold_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
                cv_scores.append(fold_rmse)

            if len(cv_scores) == 0:
                if verbose:
                    print("  Skipped: Not enough data for cross-validation")
                continue

            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            result = {
                "params": params.copy(),
                "mean_rmse": avg_score,
                "std_rmse": std_score,
                "cv_scores": cv_scores,
            }
            all_results.append(result)

            if verbose:
                print(f"  Mean RMSE: {avg_score:.6f} (+/- {std_score:.6f})")

            # Update best
            if avg_score < best_score:
                best_score = avg_score
                best_params = params.copy()
                if verbose:
                    print(f"  >>> New best score!")

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue

    print("\n" + "=" * 80)
    print("Grid Search Complete")
    print(f"Best Parameters: {best_params}")
    print(f"Best Mean RMSE: {best_score:.6f}")
    print("=" * 80)

    # Sort results by score
    all_results.sort(key=lambda x: x["mean_rmse"])

    return best_params, best_score, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Identify a linear dynamic model from reference and laser data"
    )
    parser.add_argument(
        "--ref",
        default="ref_dec_trimmed.csv",
        help="Path to reference trajectory CSV (x,y,theta)",
    )
    parser.add_argument(
        "--laser",
        default="laser_dec_trimmed.csv",
        help="Path to laser CSV (time x beams)",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Time step delta for the model (default: 1)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for identification range (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for identification range (default: length)",
    )
    parser.add_argument(
        "--save-model", default=None, help="Path to save model parameters (.npz)"
    )
    parser.add_argument(
        "--map-info",
        default="map_info.json",
        help="Path to map info JSON (for visualization)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show trajectory plot with highlighted range",
    )
    parser.add_argument(
        "--narx",
        action="store_true",
        help="Use NARX model instead of simple linear model",
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=3,
        help="Number of past time steps for NARX (default: 3)",
    )
    parser.add_argument(
        "--narx-type",
        choices=["linear", "polynomial", "neural"],
        default="linear",
        help="NARX model type: linear, polynomial, or neural network",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial degree for polynomial NARX (default: 2)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[50, 50],
        help="Hidden layer sizes for neural NARX (default: 50 50)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Regularization parameter (default: 0.01)",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Perform grid search over hyperparameters",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits for grid search (default: 5)",
    )
    parser.add_argument(
        "--grid-n-lags",
        type=int,
        nargs="+",
        default=[2, 3, 5],
        help="Grid search: n_lags values to test (default: 2 3 5)",
    )
    parser.add_argument(
        "--grid-model-types",
        type=str,
        nargs="+",
        default=["linear", "polynomial", "neural"],
        choices=["linear", "polynomial", "neural"],
        help="Grid search: model types to test (default: linear polynomial neural)",
    )
    parser.add_argument(
        "--grid-poly-degrees",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Grid search: polynomial degrees to test (default: 2 3)",
    )
    parser.add_argument(
        "--grid-alphas",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1],
        help="Grid search: alpha values to test (default: 0.001 0.01 0.1)",
    )
    parser.add_argument(
        "--grid-hidden-layers",
        type=str,
        nargs="+",
        default=["25,25", "50,50", "100,100"],
        help="Grid search: hidden layer configurations (comma-separated, default: '25,25' '50,50' '100,100')",
    )
    parser.add_argument(
        "--save-grid-results",
        default=None,
        help="Path to save grid search results JSON",
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("Step must be positive")

    ref, laser = load_series(args.ref, args.laser)

    # Parse range
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(ref)

    if start_idx < 0 or end_idx > len(ref) or start_idx >= end_idx:
        raise ValueError(
            f"Invalid range: start={start_idx}, end={end_idx}, length={len(ref)}"
        )

    ref_range = ref[start_idx:end_idx]
    laser_range = laser[start_idx:end_idx]

    if len(ref_range) <= args.step:
        raise ValueError("Step is too large for the selected range")

    # Fit model(s)
    if args.grid_search and args.narx:
        # Grid search for NARX
        print(
            f"\nPerforming grid search on range [{start_idx}, {end_idx}) with step={args.step}"
        )

        # Parse hidden layers from strings
        hidden_layers_list = []
        for hl_str in args.grid_hidden_layers:
            layers = tuple(map(int, hl_str.split(",")))
            hidden_layers_list.append(layers)

        # Build parameter grid
        param_grid = {
            "n_lags": args.grid_n_lags,
            "model_type": args.grid_model_types,
            "poly_degree": args.grid_poly_degrees,
            "hidden_layers": hidden_layers_list,
            "alpha": args.grid_alphas,
        }

        start_time = time.time()
        best_params, best_score, all_results = grid_search_narx(
            ref_range,
            laser_range,
            args.step,
            param_grid,
            n_splits=args.cv_splits,
            verbose=True,
        )
        elapsed_time = time.time() - start_time

        print(f"\nGrid search completed in {elapsed_time:.2f} seconds")

        # Print top 5 results
        print("\nTop 5 configurations:")
        print("-" * 80)
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['params']}")
            print(
                f"   Mean RMSE: {result['mean_rmse']:.6f} (+/- {result['std_rmse']:.6f})"
            )

        # Save grid search results if requested
        if args.save_grid_results:
            results_dict = {
                "best_params": best_params,
                "best_score": float(best_score),
                "all_results": [
                    {
                        "params": r["params"],
                        "mean_rmse": float(r["mean_rmse"]),
                        "std_rmse": float(r["std_rmse"]),
                        "cv_scores": [float(s) for s in r["cv_scores"]],
                    }
                    for r in all_results
                ],
                "elapsed_time": elapsed_time,
                "cv_splits": args.cv_splits,
            }
            with open(args.save_grid_results, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"\nGrid search results saved to {args.save_grid_results}")

        assert best_params is not None, "No best parameters found from grid search"

        # Train final model with best parameters
        print(f"\nTraining final model with best parameters...")
        narx_model, poly_features, rmse, narx_config = fit_narx_model(
            ref_range,
            laser_range,
            args.step,
            n_lags=best_params["n_lags"],
            model_type=best_params["model_type"],
            poly_degree=best_params.get("poly_degree", 2),
            hidden_layers=best_params.get("hidden_layers", (50, 50)),
            alpha=best_params["alpha"],
        )

        print(f"\nFinal NARX model trained on full range [{start_idx}, {end_idx})")
        print(f"Model type: {best_params['model_type']}")
        print(f"Number of lags: {best_params['n_lags']}")
        print(f"Alpha: {best_params['alpha']}")
        if best_params["model_type"] == "polynomial":
            print(f"Polynomial degree: {best_params.get('poly_degree', 2)}")
        elif best_params["model_type"] == "neural":
            print(f"Hidden layers: {best_params.get('hidden_layers', (50, 50))}")
        print(f"\nRMSE per state component [x, y, theta]:")
        print(rmse)

        # Store NARX info for saving
        A, B, bias, model = None, None, None, None

    elif args.narx:
        # NARX model without grid search
        print(f"Fitting NARX model with {args.n_lags} lags, type={args.narx_type}...")
        narx_model, poly_features, rmse, narx_config = fit_narx_model(
            ref_range,
            laser_range,
            args.step,
            n_lags=args.n_lags,
            model_type=args.narx_type,
            poly_degree=args.poly_degree,
            hidden_layers=tuple(args.hidden_layers),
            alpha=args.alpha,
        )

        print(
            f"\nFitted NARX model on range [{start_idx}, {end_idx}) with step={args.step}"
        )
        print(f"Model type: {args.narx_type}")
        print(f"Number of lags: {args.n_lags}")
        print(f"Input features: {narx_config['n_features']}")
        if args.narx_type == "polynomial":
            print(f"Polynomial degree: {args.poly_degree}")
        elif args.narx_type == "neural":
            print(f"Hidden layers: {args.hidden_layers}")
        print(f"\nRMSE per state component [x, y, theta]:")
        print(rmse)

        # Store NARX info for saving
        A, B, bias, model = None, None, None, None
    else:
        # Simple linear model
        A, B, bias, rmse, model = fit_state_space(ref_range, laser_range, args.step)
        narx_model, poly_features, narx_config = None, None, None

        print(
            f"Fitted linear model on range [{start_idx}, {end_idx}) with step={args.step}"
        )
        print("Fitted linear model: x_{k+1} = A x_k + B f(laser_k) + bias")
        print("A matrix (3x3):")
        print(A)
        print(
            "\nB matrix (3 x features): features = [mean, min, std, left_mean, front_mean, right_mean]"
        )
        print(B)
        print("\nBias:")
        print(bias)
        print("\nRMSE per state component [x, y, theta]:")
        print(rmse)

    if args.save_model:
        if args.narx:
            # Save NARX model
            import pickle

            save_dict = {
                "model_type": "narx",
                "narx_config": narx_config,
                "rmse": rmse.tolist(),
            }

            # Save as JSON (config only)
            with open(args.save_model + ".json", "w") as f:
                json.dump(save_dict, f, indent=4)

            # Save full model with pickle (includes sklearn objects)
            model_data = {
                "narx_model": narx_model,
                "poly_features": poly_features,
                "narx_config": narx_config,
                "rmse": rmse,
            }
            with open(args.save_model + ".pkl", "wb") as f:
                pickle.dump(model_data, f)

            print(
                f"NARX model saved to {args.save_model}.pkl and {args.save_model}.json"
            )
        else:
            # Save simple linear model
            assert (
                A is not None
                and B is not None
                and bias is not None
                and model is not None
            )
            np.savez(
                args.save_model + ".npz",
                A=A,
                B=B,
                bias=bias,
                rmse=rmse,
                coef=model.coef_,
                intercept=model.intercept_,
            )

            model_dict = {
                "A": A.tolist(),
                "B": B.tolist(),
                "bias": bias.tolist() if type(bias) is np.ndarray else bias,
                "rmse": rmse.tolist(),
                "coef": model.coef_.tolist(),
                "intercept": (
                    model.intercept_.tolist()
                    if type(model.intercept_) is np.ndarray
                    else model.intercept_
                ),
            }

            with open(args.save_model + ".json", "w") as f:
                json.dump(model_dict, f, indent=4)

            print(f"Model saved to {args.save_model}.npz and {args.save_model}.json")

    if args.show_plot:
        # Load map info
        try:
            with open(args.map_info, "r") as f:
                map_info = json.load(f)
            map_img = Image.open(map_info["image"])

            fig, ax = plt.subplots(figsize=(12, 10))

            extent = (
                map_info["xlimits"][0],
                map_info["xlimits"][1],
                map_info["ylimits"][0],
                map_info["ylimits"][1],
            )
            ax.imshow(map_img, extent=extent)

            # Plot full trajectory in light gray
            ax.plot(
                ref[:, 0],
                ref[:, 1],
                color="lightgray",
                linewidth=1,
                label="Full trajectory",
            )

            # Highlight selected range
            ax.plot(
                ref_range[:, 0],
                ref_range[:, 1],
                "b-",
                linewidth=2,
                label=f"Selected range [{start_idx}, {end_idx})",
            )

            # Mark range boundaries
            ax.plot(
                ref_range[0, 0],
                ref_range[0, 1],
                "go",
                markersize=10,
                label="Range start",
            )
            ax.plot(
                ref_range[-1, 0],
                ref_range[-1, 1],
                "ro",
                markersize=10,
                label="Range end",
            )

            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            title = (
                f"Reference Trajectory - Identification Range [{start_idx}, {end_idx})"
            )
            if args.narx:
                title += f" (NARX: {args.narx_type}, lags={args.n_lags})"
            ax.set_title(title)
            ax.set_xlim(map_info["xlimits"])
            ax.set_ylim(map_info["ylimits"])
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            plt.tight_layout()
            plt.show()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load map for visualization: {e}")


if __name__ == "__main__":
    main()
