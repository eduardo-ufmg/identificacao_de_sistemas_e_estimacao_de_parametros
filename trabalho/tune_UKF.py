import argparse
import json

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from LaserDynamicModel import load_model as load_laser_model
from OdometryDynamicModel import load_model as load_odo_model
from UKF import UKFEstimator, build_odometry_trajectory, load_laser_data


def load_data(
    odo_diff_path: str,
    laser_path: str,
    model_path: str,
    map_info_path: str,
    odo_model_path: str | None = None,
):
    """Load and prepare all data."""
    with open(map_info_path, "r") as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info["initial_pose"], dtype=float)

    odo_deltas = np.loadtxt(odo_diff_path, delimiter=",")
    if odo_deltas.ndim == 1:
        odo_deltas = odo_deltas.reshape(-1, 3)
    laser_data = load_laser_data(laser_path)
    laser_model_params = load_laser_model(model_path)

    odo_model_params = None
    if odo_model_path:
        odo_model_params = load_odo_model(odo_model_path)

    n_steps = min(len(odo_deltas), len(laser_data))
    odometry = odo_deltas[:n_steps]
    laser_scans = laser_data[:n_steps]

    return initial_pose, odometry, laser_scans, laser_model_params, odo_model_params


def create_objective_function(
    initial_pose,
    odo_deltas,
    laser_scans,
    laser_model_params,
    odo_model_params,
    ground_truth,
):
    """
    Create objective function closure for Bayesian Optimization.

    Returns a function that takes individual parameters and returns RMSE.
    """

    def objective(q_x, q_y, q_theta, r_x, r_y, r_theta):
        """
        Objective function for Bayesian Optimization.
        Returns RMSE between estimated and ground truth trajectories.
        """
        q_std = (q_x, q_y, q_theta)
        r_std = (r_x, r_y, r_theta)

        try:
            estimator = UKFEstimator(initial_pose, q_std=q_std, r_std=r_std)
            est_states = estimator.run(
                odo_deltas, laser_scans, laser_model_params, odo_model_params
            )

            # Compare to ground truth trajectory
            n_steps = min(len(est_states), len(ground_truth))
            error = est_states[:n_steps] - ground_truth[:n_steps]
            rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))

            return rmse
        except Exception as e:
            print(f"  Error in evaluation: {e}")
            return 1e10

    return objective


def main():
    parser = argparse.ArgumentParser(description="Tune UKF parameters via optimization")
    parser.add_argument(
        "--odo-diff",
        default="odo_dec_diff_trimmed.csv",
        help="Path to odometry differences CSV",
    )
    parser.add_argument(
        "--laser", default="laser_dec_trimmed.csv", help="Path to laser data CSV"
    )
    parser.add_argument(
        "--model", default="laser_model.json", help="Path to laser model JSON"
    )
    parser.add_argument(
        "--odo-model", default=None, help="Path to odometry NARX model JSON (optional)"
    )
    parser.add_argument(
        "--map-info", default="map_info.json", help="Path to map info JSON"
    )
    parser.add_argument(
        "--ground-truth", default=None, help="Path to ground truth trajectory CSV"
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=50,
        help="Number of Bayesian optimization iterations",
    )
    parser.add_argument(
        "--n-initial", type=int, default=10, help="Number of random initial evaluations"
    )
    parser.add_argument(
        "--q-bounds",
        type=float,
        nargs=2,
        default=[0.01, 1.0],
        help="Search bounds for Q std parameters (min max)",
    )
    parser.add_argument(
        "--r-bounds",
        type=float,
        nargs=2,
        default=[0.01, 1.0],
        help="Search bounds for R std parameters (min max)",
    )
    parser.add_argument(
        "--output", default="ukf_tuned.json", help="Path to save tuned parameters"
    )
    parser.add_argument(
        "--random-state", type=int, default=0, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Load data
    initial_pose, odo_deltas, laser_scans, laser_model_params, odo_model_params = (
        load_data(args.odo_diff, args.laser, args.model, args.map_info, args.odo_model)
    )

    # Check model type
    is_narx = laser_model_params[0]
    if is_narx:
        _, _, _, narx_config = laser_model_params
        print(
            f"Using NARX model: {narx_config['model_type']}, n_lags={narx_config['n_lags']}"
        )
    else:
        print("Using linear model")

    # Load ground truth trajectory (required for meaningful optimization)
    if args.ground_truth:
        try:
            ground_truth = np.loadtxt(args.ground_truth, delimiter=",")
            print(
                f"Loaded ground truth trajectory with {len(ground_truth)} steps for evaluation."
            )
        except Exception as e:
            print(f"Error: Could not load ground truth trajectory: {e}")
            print(
                "Ground truth is required for tuning with one-step-ahead predictions."
            )
            return
    else:
        print("Error: --ground-truth argument is required for UKF tuning.")
        print("Example: --ground-truth ref_dec.csv")
        return

    # Define search space for Bayesian Optimization
    search_space = [
        Real(args.q_bounds[0], args.q_bounds[1], name="q_x"),
        Real(args.q_bounds[0], args.q_bounds[1], name="q_y"),
        Real(args.q_bounds[0], args.q_bounds[1], name="q_theta"),
        Real(args.r_bounds[0], args.r_bounds[1], name="r_x"),
        Real(args.r_bounds[0], args.r_bounds[1], name="r_y"),
        Real(args.r_bounds[0], args.r_bounds[1], name="r_theta"),
    ]

    print(f"\nStarting Bayesian Optimization")
    print(f"Number of iterations: {args.n_calls}")
    print(f"Initial random evaluations: {args.n_initial}")
    print(f"Q bounds: {args.q_bounds}")
    print(f"R bounds: {args.r_bounds}")
    print(f"Random state: {args.random_state}")

    # Create objective function
    objective_fn = create_objective_function(
        initial_pose,
        odo_deltas,
        laser_scans,
        laser_model_params,
        odo_model_params,
        ground_truth,
    )

    # Decorate with use_named_args for named parameter passing
    @use_named_args(search_space)
    def objective_wrapper(**params):
        return objective_fn(**params)

    # Run Bayesian Optimization
    print("\nRunning optimization...")
    result = gp_minimize(
        objective_wrapper,
        search_space,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        random_state=args.random_state,
        verbose=False,
        n_jobs=-1,
    )

    assert result is not None, "Optimization failed, no result returned."
    q_opt = result.x[:3]
    r_opt = result.x[3:]

    print(f"\nOptimization complete!")
    print(f"Total evaluations: {len(result.func_vals)}")
    print(f"Best objective value (RMSE): {result.fun:.6e}")
    print(f"Optimized Q std: {q_opt}")
    print(f"Optimized R std: {r_opt}")

    # Save parameters
    params = {
        "q_std": [float(x) for x in q_opt],
        "r_std": [float(x) for x in r_opt],
        "objective_value": float(result.fun),
        "method": "Bayesian Optimization (GP)",
        "n_calls": args.n_calls,
        "n_initial_points": args.n_initial,
        "total_evaluations": len(result.func_vals),
        "q_bounds": args.q_bounds,
        "r_bounds": args.r_bounds,
        "random_state": args.random_state,
    }

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nTuned parameters saved to {args.output}")


if __name__ == "__main__":
    main()
