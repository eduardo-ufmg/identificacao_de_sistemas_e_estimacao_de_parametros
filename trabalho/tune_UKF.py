import argparse
import json

import numpy as np
from scipy.optimize import minimize

from UKF import (
    UKFEstimator,
    build_laser_measurements,
    build_odometry_trajectory,
    load_model,
)


def load_data(odo_diff_path: str, laser_path: str, model_path: str, map_info_path: str):
    """Load and prepare all data."""
    with open(map_info_path, "r") as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info["initial_pose"], dtype=float)

    odo_diff = np.loadtxt(odo_diff_path, delimiter=",")
    if odo_diff.ndim == 1:
        odo_diff = odo_diff.reshape(-1, 3)
    laser_traj = build_laser_measurements(laser_path, model_path, initial_pose)

    n_steps = min(len(odo_diff), len(laser_traj) - 1)
    odometry = odo_diff[:n_steps]
    laser_meas = laser_traj[1 : n_steps + 1]

    return initial_pose, odometry, laser_meas


def objective(params, initial_pose, odometry, laser_meas, ground_truth=None):
    """
    Objective function for optimization.
    params = [q_x, q_y, q_theta, r_x, r_y, r_theta]

    If ground_truth is provided, compares estimated trajectory to it.
    Otherwise, penalizes deviation from laser measurements.
    """
    q_std = params[:3]
    r_std = params[3:]

    if np.any(q_std <= 0) or np.any(r_std <= 0):
        return 1e10

    estimator = UKFEstimator(initial_pose, q_std=tuple(q_std), r_std=tuple(r_std))
    est_states = estimator.run(odometry, laser_meas)[:-1]

    if ground_truth is not None:
        # Compare to ground truth (if available)
        error = est_states - ground_truth
        error_mean = np.mean(error**2)
        error_std = np.std(error)
        error_function = error_mean + error_std
    else:
        # Penalize difference between initial and final positions as proxy
        est_initial = est_states[0]
        est_final = est_states[-1]
        position_diff = np.linalg.norm(est_final[:2] - est_initial[:2])
        error_function = position_diff

    return error_function


def main():
    parser = argparse.ArgumentParser(description="Tune UKF parameters via optimization")
    parser.add_argument(
        "--odo-diff",
        default="odo_dec_diff.csv",
        help="Path to odometry differences CSV",
    )
    parser.add_argument(
        "--laser", default="laser_dec.csv", help="Path to laser data CSV"
    )
    parser.add_argument(
        "--model", default="laser_model.json", help="Path to laser model JSON"
    )
    parser.add_argument(
        "--map-info", default="map_info.json", help="Path to map info JSON"
    )
    parser.add_argument(
        "--ground-truth", default=None, help="Path to ground truth trajectory CSV"
    )
    parser.add_argument(
        "--init-q-std", type=float, nargs=3, default=[1, 1, 1], help="Initial Q std"
    )
    parser.add_argument(
        "--init-r-std", type=float, nargs=3, default=[1, 1, 1], help="Initial R std"
    )
    parser.add_argument(
        "--method",
        default="Nelder-Mead",
        help="Optimization method (Nelder-Mead, Powell, BFGS, etc.)",
    )
    parser.add_argument(
        "--output", default="ukf_tuned.json", help="Path to save tuned parameters"
    )
    args = parser.parse_args()

    # Load data
    initial_pose, odometry, laser_meas = load_data(
        args.odo_diff, args.laser, args.model, args.map_info
    )

    # Initial guess
    x0 = np.concatenate([args.init_q_std, args.init_r_std])

    print(f"Starting tuning with method: {args.method}")
    print(f"Initial Q std: {args.init_q_std}")
    print(f"Initial R std: {args.init_r_std}")

    # Load ground truth trajectory
    if args.ground_truth:
        try:
            ground_truth = np.loadtxt(args.ground_truth, delimiter=",")
            print(
                f"Loaded ground truth trajectory with {len(ground_truth)} steps for evaluation."
            )
        except Exception as e:
            print(f"Could not load ground truth trajectory: {e}")
            ground_truth = None
    else:
        ground_truth = None

    # Optimize
    result = minimize(
        objective,
        x0,
        args=(initial_pose, odometry, laser_meas, ground_truth),
        method=args.method,
    )

    if not result.success:
        print(f"Warning: optimization may not have converged: {result.message}")

    q_opt = result.x[:3]
    r_opt = result.x[3:]

    print(f"\nOptimization complete (iterations: {result.nit})")
    print(f"Final objective value: {result.fun:.6e}")
    print(f"Optimized Q std: {q_opt}")
    print(f"Optimized R std: {r_opt}")

    # Save parameters
    params = {
        "q_std": q_opt.tolist(),
        "r_std": r_opt.tolist(),
        "objective_value": float(result.fun),
        "method": args.method,
        "iterations": int(result.nit),
    }

    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nTuned parameters saved to {args.output}")


if __name__ == "__main__":
    main()
