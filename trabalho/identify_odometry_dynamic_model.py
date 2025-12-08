import argparse
import json
import pickle
import time
from itertools import product

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures


def load_data(ref_path: str, odo_diff_path: str):
    """Load reference trajectory and odometry differences."""
    ref = np.loadtxt(ref_path, delimiter=",")
    odo_diff = np.loadtxt(odo_diff_path, delimiter=",")

    if ref.ndim != 2 or ref.shape[1] < 3:
        raise ValueError("Reference data must have at least 3 columns (x, y, theta)")
    if odo_diff.ndim == 1:
        odo_diff = odo_diff.reshape(-1, 3)

    length = min(len(ref), len(odo_diff))
    if length < 2:
        raise ValueError("Not enough samples to fit a dynamic model")

    return ref[:length], odo_diff[:length]


def fit_narx_model(
    ref: np.ndarray,
    odo_diff: np.ndarray,
    step: int = 1,
    n_lags: int = 3,
    model_type: str = "linear",
    poly_degree: int = 2,
    hidden_layers: tuple = (50, 50),
    alpha: float = 0.01,
):
    """
    Fit a NARX (Nonlinear AutoRegressive with eXogenous inputs) model for odometry.

    Args:
        ref: Reference trajectory (n_samples, 3) with [x, y, theta]
        odo_diff: Odometry differences (n_samples, 3) with [dx, dy, dtheta]
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

    # Build NARX input features: past states and current odometry delta
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

        # Current odometry delta at time i+n_lags-1
        current_odo = odo_diff[i + n_lags - 1]  # Shape: (3,)

        # Concatenate
        x_i = np.concatenate([past_states, current_odo])
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
            alpha=alpha,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )
        model.fit(X, y)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute RMSE
    if model_type == "polynomial" and poly_features is not None:
        X_pred = poly_features.transform(X)
    else:
        X_pred = X

    pred = model.predict(X_pred)
    rmse = np.sqrt(np.mean((pred - y) ** 2, axis=0))

    # Store configuration
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
    odo_diff: np.ndarray,
    n_lags_list: list,
    model_types: list,
    poly_degrees: list,
    alphas: list,
    hidden_layers_list: list,
    cv_splits: int = 5,
):
    """
    Grid search over NARX hyperparameters with time series cross-validation.

    Args:
        ref: Reference trajectory
        odo_diff: Odometry differences
        n_lags_list: List of n_lags values to try
        model_types: List of model types ('linear', 'polynomial', 'neural')
        poly_degrees: List of polynomial degrees
        alphas: List of alpha values
        hidden_layers_list: List of hidden layer configurations
        cv_splits: Number of cross-validation splits

    Returns:
        results: List of dicts with parameters and CV scores
        best_params: Dictionary with best hyperparameters
    """
    states = ref[:, :3]

    results = []
    best_score = float("inf")
    best_params = None

    # Create all parameter combinations
    param_combinations = []
    for n_lags in n_lags_list:
        for model_type in model_types:
            for alpha in alphas:
                if model_type == "polynomial":
                    for poly_degree in poly_degrees:
                        param_combinations.append(
                            {
                                "n_lags": n_lags,
                                "model_type": model_type,
                                "poly_degree": poly_degree,
                                "alpha": alpha,
                                "hidden_layers": None,
                            }
                        )
                elif model_type == "neural":
                    for hidden_layers in hidden_layers_list:
                        param_combinations.append(
                            {
                                "n_lags": n_lags,
                                "model_type": model_type,
                                "poly_degree": None,
                                "alpha": alpha,
                                "hidden_layers": hidden_layers,
                            }
                        )
                else:  # linear
                    param_combinations.append(
                        {
                            "n_lags": n_lags,
                            "model_type": model_type,
                            "poly_degree": None,
                            "alpha": alpha,
                            "hidden_layers": None,
                        }
                    )

    print(f"\nGrid search: testing {len(param_combinations)} parameter combinations")
    print(f"Using {cv_splits}-fold time series cross-validation\n")

    for idx, params in enumerate(param_combinations, 1):
        print(f"[{idx}/{len(param_combinations)}] Testing: {params}")

        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            cv_scores = []

            for train_idx, val_idx in tscv.split(states):
                # Split data
                ref_train = ref[train_idx]
                odo_train = odo_diff[train_idx]
                ref_val = ref[val_idx]
                odo_val = odo_diff[val_idx]

                # Fit model
                model, poly_features, _, config = fit_narx_model(
                    ref_train,
                    odo_train,
                    step=1,
                    n_lags=params["n_lags"],
                    model_type=params["model_type"],
                    poly_degree=params["poly_degree"] or 2,
                    hidden_layers=(
                        tuple(params["hidden_layers"])
                        if params["hidden_layers"]
                        else (50, 50)
                    ),
                    alpha=params["alpha"],
                )

                # Evaluate on validation set
                states_val = ref_val[:, :3]
                n_val_samples = len(states_val) - params["n_lags"]

                if n_val_samples < 1:
                    continue

                X_val_list = []
                y_val_list = []

                for i in range(n_val_samples):
                    past_states = states_val[i : i + params["n_lags"]].flatten()
                    current_odo = odo_val[i + params["n_lags"] - 1]
                    x_i = np.concatenate([past_states, current_odo])
                    X_val_list.append(x_i)
                    y_val_list.append(states_val[i + params["n_lags"]])

                X_val = np.array(X_val_list)
                y_val = np.array(y_val_list)

                if params["model_type"] == "polynomial" and poly_features is not None:
                    X_val = poly_features.transform(X_val)

                pred_val = model.predict(X_val)
                val_rmse = np.sqrt(np.mean((pred_val - y_val) ** 2))
                cv_scores.append(val_rmse)

            if len(cv_scores) > 0:
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)

                result = {
                    "params": params.copy(),
                    "cv_mean_rmse": mean_cv_score,
                    "cv_std_rmse": std_cv_score,
                    "cv_scores": cv_scores,
                }
                results.append(result)

                print(f"  CV RMSE: {mean_cv_score:.6f} ± {std_cv_score:.6f}")

                if mean_cv_score < best_score:
                    best_score = mean_cv_score
                    best_params = params.copy()
                    print(f"  *** New best score: {best_score:.6f}")
            else:
                print(f"  Skipped (insufficient validation samples)")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Sort results by CV score
    results.sort(key=lambda x: x["cv_mean_rmse"])

    return results, best_params


def main():
    parser = argparse.ArgumentParser(description="Identify odometry NARX dynamic model")
    parser.add_argument(
        "--reference",
        default="ref_dec_trimmed.csv",
        help="Path to reference trajectory CSV",
    )
    parser.add_argument(
        "--odo-diff",
        default="odo_dec_diff_trimmed.csv",
        help="Path to odometry differences CSV",
    )
    parser.add_argument(
        "--narx", action="store_true", help="Fit NARX model instead of linear model"
    )
    parser.add_argument(
        "--n-lags", type=int, default=3, help="Number of past states to use (NARX)"
    )
    parser.add_argument(
        "--narx-type",
        choices=["linear", "polynomial", "neural"],
        default="linear",
        help="Type of NARX model",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="Polynomial degree (for polynomial NARX)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[50, 50],
        help="Hidden layer sizes (for neural NARX)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Regularization parameter"
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Path prefix to save model files (.json and .pkl)",
    )

    # Grid search arguments
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Perform grid search over hyperparameters",
    )
    parser.add_argument(
        "--cv-splits", type=int, default=5, help="Number of cross-validation splits"
    )
    parser.add_argument(
        "--grid-n-lags",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
        help="List of n_lags values for grid search",
    )
    parser.add_argument(
        "--grid-model-types",
        choices=["linear", "polynomial", "neural"],
        nargs="+",
        default=["linear", "polynomial", "neural"],
        help="Model types for grid search",
    )
    parser.add_argument(
        "--grid-poly-degrees",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Polynomial degrees for grid search",
    )
    parser.add_argument(
        "--grid-alphas",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="Alpha values for grid search",
    )
    parser.add_argument(
        "--grid-hidden-layers",
        type=str,
        nargs="+",
        default=["50,50", "100,50", "25,25,25"],
        help="Hidden layer configs for grid search (comma-separated)",
    )
    parser.add_argument(
        "--save-grid-results",
        default=None,
        help="Path to save grid search results JSON",
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    ref, odo_diff = load_data(args.reference, args.odo_diff)
    print(f"Loaded {len(ref)} samples")

    if args.grid_search:
        # Parse hidden layer configurations
        hidden_layers_list = []
        for config_str in args.grid_hidden_layers:
            layers = tuple(int(x) for x in config_str.split(","))
            hidden_layers_list.append(layers)

        # Run grid search
        results, best_params = grid_search_narx(
            ref,
            odo_diff,
            n_lags_list=args.grid_n_lags,
            model_types=args.grid_model_types,
            poly_degrees=args.grid_poly_degrees,
            alphas=args.grid_alphas,
            hidden_layers_list=hidden_layers_list,
            cv_splits=args.cv_splits,
        )

        print("\n" + "=" * 80)
        print("GRID SEARCH RESULTS")
        print("=" * 80)
        print(f"\nTop 5 configurations:")
        for i, result in enumerate(results[:5], 1):
            print(
                f"\n{i}. CV RMSE: {result['cv_mean_rmse']:.6f} ± {result['cv_std_rmse']:.6f}"
            )
            print(f"   Parameters: {result['params']}")

        print(f"\nBest parameters: {best_params}")

        # Save grid search results
        if args.save_grid_results:
            with open(args.save_grid_results, "w") as f:
                json.dump({"results": results, "best_params": best_params}, f, indent=2)
            print(f"\nGrid search results saved to: {args.save_grid_results}")

        # Fit final model with best parameters
        if args.save_model and best_params:
            print(f"\nFitting final model with best parameters...")
            model, poly_features, rmse, config = fit_narx_model(
                ref,
                odo_diff,
                step=1,
                n_lags=best_params["n_lags"],
                model_type=best_params["model_type"],
                poly_degree=best_params["poly_degree"] or 2,
                hidden_layers=(
                    tuple(best_params["hidden_layers"])
                    if best_params["hidden_layers"]
                    else (50, 50)
                ),
                alpha=best_params["alpha"],
            )

            # Save model
            json_path = f"{args.save_model}.json"
            pkl_path = f"{args.save_model}.pkl"

            # Save config to JSON
            json_data = {
                "model_type": "narx",
                "narx_config": config,
                "rmse": rmse.tolist(),
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

            # Save model to pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {
                        "narx_model": model,
                        "poly_features": poly_features,
                        "narx_config": config,
                    },
                    f,
                )

            print(f"\nModel saved:")
            print(f"  Config: {json_path}")
            print(f"  Weights: {pkl_path}")
            print(f"  RMSE: {rmse}")

    elif args.narx:
        # Fit single NARX model
        print(f"\nFitting NARX model (type={args.narx_type}, n_lags={args.n_lags})...")

        model, poly_features, rmse, config = fit_narx_model(
            ref,
            odo_diff,
            step=1,
            n_lags=args.n_lags,
            model_type=args.narx_type,
            poly_degree=args.poly_degree,
            hidden_layers=tuple(args.hidden_layers),
            alpha=args.alpha,
        )

        print(f"\nNARX Model fitted successfully!")
        print(f"RMSE per component: {rmse}")
        print(f"Overall RMSE: {np.linalg.norm(rmse):.6f}")

        # Save model if requested
        if args.save_model:
            json_path = f"{args.save_model}.json"
            pkl_path = f"{args.save_model}.pkl"

            # Save config to JSON
            json_data = {
                "model_type": "narx",
                "narx_config": config,
                "rmse": rmse.tolist(),
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

            # Save model to pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {
                        "narx_model": model,
                        "poly_features": poly_features,
                        "narx_config": config,
                    },
                    f,
                )

            print(f"\nModel saved:")
            print(f"  Config: {json_path}")
            print(f"  Weights: {pkl_path}")

    else:
        print(
            "\nPlease use --narx flag to fit NARX model or --grid-search for hyperparameter optimization"
        )
        print(
            "Example: python3 identify_odometry_dynamic_model.py --narx --save-model odo_narx"
        )
        print(
            "Example: python3 identify_odometry_dynamic_model.py --grid-search --save-model best_odo_narx"
        )


if __name__ == "__main__":
    main()
