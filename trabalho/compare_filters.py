#!/usr/bin/env python3
"""
Compare UKF, EKF, and Particle Filter performance on robot localization.

This script runs all three filters with the same data and parameters,
computes statistical metrics, and generates comparison plots.
"""

import argparse
import json
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

from UKF import UKFEstimator, load_laser_data, build_odometry_trajectory
from EKF import EKFEstimator
from PF import ParticleFilter
from LaserDynamicModel import load_model as load_laser_model
from OdometryDynamicModel import load_model as load_odo_model


def load_data(args):
    """Load all required data files."""
    # Map info
    with open(args.map_info, "r") as f:
        map_info = json.load(f)
    map_img = Image.open(map_info["image"])
    initial_pose = np.array(map_info["initial_pose"], dtype=float)
    
    # Odometry deltas
    odo_deltas = np.loadtxt(args.odo_diff, delimiter=",")
    if odo_deltas.ndim == 1:
        odo_deltas = odo_deltas.reshape(-1, 3)
    
    # Laser data
    laser_data = load_laser_data(args.laser)
    
    # Laser model
    laser_model_params = load_laser_model(args.model)
    
    # Odometry model (optional)
    odo_model_params = None
    if args.odo_model:
        odo_model_params = load_odo_model(args.odo_model)
    
    # Reference trajectory (ground truth)
    ref_traj = None
    if args.reference:
        ref_traj = np.loadtxt(args.reference, delimiter=",")
        if ref_traj.ndim == 1:
            ref_traj = ref_traj.reshape(-1, 3)
    
    # Odometry trajectory for comparison - use NARX model if provided
    if odo_model_params is not None:
        # Use odometry NARX model
        from OdometryDynamicModel import OdometryNARXModel
        is_odo_narx, odo_narx_model, odo_poly_features, odo_narx_config = odo_model_params
        odo_model_obj = OdometryNARXModel(odo_narx_model, odo_poly_features, odo_narx_config, initial_pose)
        odom_traj = odo_model_obj.simulate(odo_deltas)
    else:
        # Use simple additive odometry model
        odom_traj = build_odometry_trajectory(args.odo_diff, initial_pose)
    
    return {
        'map_info': map_info,
        'map_img': map_img,
        'initial_pose': initial_pose,
        'odo_deltas': odo_deltas,
        'laser_data': laser_data,
        'laser_model_params': laser_model_params,
        'odo_model_params': odo_model_params,
        'ref_traj': ref_traj,
        'odom_traj': odom_traj,
    }


def run_filter(filter_class, filter_name: str, data: dict, params: dict, verbose: bool = True) -> Tuple[np.ndarray, float, dict]:
    """
    Run a filter and return trajectory, execution time, and additional stats.
    
    Returns:
        trajectory: (N, 3) array of estimated states
        exec_time: execution time in seconds
        stats: dictionary with filter-specific statistics
    """
    if verbose:
        print(f"\nRunning {filter_name}...")
    
    initial_pose = data['initial_pose']
    odo_deltas = data['odo_deltas']
    laser_data = data['laser_data']
    laser_model_params = data['laser_model_params']
    odo_model_params = data.get('odo_model_params')
    
    # Create filter instance
    if filter_name == "UKF":
        filter_obj = UKFEstimator(
            initial_pose, 
            q_std=params['q_std'], 
            r_std=params['r_std']
        )
    elif filter_name == "EKF":
        filter_obj = EKFEstimator(
            initial_pose, 
            q_std=params['q_std'], 
            r_std=params['r_std']
        )
    elif filter_name == "PF":
        filter_obj = ParticleFilter(
            initial_pose, 
            n_particles=params.get('n_particles', 1000),
            q_std=params['q_std'], 
            r_std=params['r_std'],
            resample_threshold=params.get('resample_threshold', 0.5)
        )
    else:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    # Run filter with timing
    start_time = time.time()
    trajectory = filter_obj.run(odo_deltas, laser_data, laser_model_params, odo_model_params)
    exec_time = time.time() - start_time
    
    # Collect filter-specific stats
    filter_stats = {}
    if filter_name == "PF":
        filter_stats['n_particles'] = params.get('n_particles', 1000)
        filter_stats['resample_threshold'] = params.get('resample_threshold', 0.5)
    
    if verbose:
        print(f"  Execution time: {exec_time:.4f} seconds")
        print(f"  Trajectory length: {len(trajectory)} steps")
    
    return trajectory, exec_time, filter_stats


def compute_metrics(estimated: np.ndarray, reference: np.ndarray, name: str) -> Dict:
    """
    Compute error metrics between estimated and reference trajectories.
    
    Args:
        estimated: (N, 3) array [x, y, theta]
        reference: (N, 3) array [x, y, theta]
        name: filter name for display
    
    Returns:
        Dictionary with metric names and values
    """
    n_steps = min(len(estimated), len(reference))
    estimated = estimated[:n_steps]
    reference = reference[:n_steps]
    
    # Position errors (x, y)
    pos_error = estimated[:, :2] - reference[:, :2]
    pos_distances = np.linalg.norm(pos_error, axis=1)
    
    # Orientation errors (theta)
    theta_error = estimated[:, 2] - reference[:, 2]
    # Normalize to [-pi, pi]
    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    
    # Compute metrics
    metrics = {
        'filter': name,
        # Position metrics
        'pos_rmse': np.sqrt(np.mean(pos_distances**2)),
        'pos_mean': np.mean(pos_distances),
        'pos_std': np.std(pos_distances),
        'pos_median': np.median(pos_distances),
        'pos_max': np.max(pos_distances),
        'pos_min': np.min(pos_distances),
        'pos_q95': np.percentile(pos_distances, 95),
        # Orientation metrics (in degrees)
        'theta_rmse': np.sqrt(np.mean(theta_error**2)) * 180/np.pi,
        'theta_mean': np.mean(np.abs(theta_error)) * 180/np.pi,
        'theta_std': np.std(theta_error) * 180/np.pi,
        'theta_median': np.median(np.abs(theta_error)) * 180/np.pi,
        'theta_max': np.max(np.abs(theta_error)) * 180/np.pi,
        # Component-wise RMSE
        'x_rmse': np.sqrt(np.mean(pos_error[:, 0]**2)),
        'y_rmse': np.sqrt(np.mean(pos_error[:, 1]**2)),
    }
    
    return metrics


def print_metrics_table(all_metrics: list):
    """Print formatted table of metrics for all filters."""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*80)
    
    # Position metrics
    print("\nPosition Error Metrics (meters):")
    print("-" * 80)
    print(f"{'Filter':<10} {'RMSE':>10} {'Mean':>10} {'Median':>10} {'Std':>10} {'Max':>10} {'95%ile':>10}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['filter']:<10} {m['pos_rmse']:>10.4f} {m['pos_mean']:>10.4f} "
              f"{m['pos_median']:>10.4f} {m['pos_std']:>10.4f} {m['pos_max']:>10.4f} "
              f"{m['pos_q95']:>10.4f}")
    
    # Orientation metrics
    print("\nOrientation Error Metrics (degrees):")
    print("-" * 80)
    print(f"{'Filter':<10} {'RMSE':>10} {'Mean':>10} {'Median':>10} {'Std':>10} {'Max':>10}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['filter']:<10} {m['theta_rmse']:>10.2f} {m['theta_mean']:>10.2f} "
              f"{m['theta_median']:>10.2f} {m['theta_std']:>10.2f} {m['theta_max']:>10.2f}")
    
    # Component-wise RMSE
    print("\nComponent-wise RMSE (meters):")
    print("-" * 40)
    print(f"{'Filter':<10} {'X-RMSE':>12} {'Y-RMSE':>12}")
    print("-" * 40)
    for m in all_metrics:
        print(f"{m['filter']:<10} {m['x_rmse']:>12.4f} {m['y_rmse']:>12.4f}")
    print("="*80)


def statistical_tests(trajectories: dict, reference: np.ndarray):
    """Perform statistical significance tests between filters."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    filters = list(trajectories.keys())
    n_steps = min([len(traj) for traj in trajectories.values()] + [len(reference)])
    
    # Compute position errors for each filter
    errors = {}
    for name, traj in trajectories.items():
        pos_error = traj[:n_steps, :2] - reference[:n_steps, :2]
        errors[name] = np.linalg.norm(pos_error, axis=1)
    
    # Pairwise comparisons using Wilcoxon signed-rank test
    print("\nPairwise Wilcoxon Signed-Rank Tests (Position Errors):")
    print("-" * 80)
    print("Null hypothesis: The two filters have the same error distribution")
    print("Alternative: The errors are different")
    print()
    
    for i, filter1 in enumerate(filters):
        for filter2 in filters[i+1:]:
            wilcoxon_results = stats.wilcoxon(errors[filter1], errors[filter2])
            p_value = wilcoxon_results.pvalue # type: ignore
            statistic = wilcoxon_results.statistic # type: ignore
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
            print(f"{filter1:>6} vs {filter2:<6}: p-value = {p_value:.4e} {significance}")
    
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
    
    # Friedman test (non-parametric repeated measures ANOVA)
    print("\n" + "-" * 80)
    print("Friedman Test (Overall comparison):")
    error_matrix = np.column_stack([errors[name] for name in filters])
    statistic, p_value = stats.friedmanchisquare(*[error_matrix[:, i] for i in range(len(filters))])
    print(f"Chi-square statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4e}")
    if p_value < 0.05:
        print("Conclusion: At least one filter performs significantly differently from others")
    else:
        print("Conclusion: No significant difference between filters")
    print("="*80)


def plot_comparison(data: dict, trajectories: dict, exec_times: dict, output_path: str):
    """Create comprehensive comparison plot and save to file."""
    map_info = data['map_info']
    map_img = data['map_img']
    ref_traj = data['ref_traj']
    odom_traj = data['odom_traj']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main trajectory plot
    ax_main = fig.add_subplot(gs[:, :2])
    extent = (
        map_info["xlimits"][0],
        map_info["xlimits"][1],
        map_info["ylimits"][0],
        map_info["ylimits"][1],
    )
    ax_main.imshow(map_img, extent=extent, alpha=0.7)
    
    # Plot trajectories
    colors = {'UKF': 'lime', 'EKF': 'red', 'PF': 'orange'}
    linestyles = {'UKF': '-', 'EKF': '-', 'PF': '-'}
    linewidths = {'UKF': 2, 'EKF': 2, 'PF': 2}
    
    if ref_traj is not None:
        ax_main.plot(ref_traj[:, 0], ref_traj[:, 1], 'k-', linewidth=2.5, 
                    label='Reference (Ground Truth)', alpha=0.8)
    
    # Label odometry trajectory based on model type
    odo_label = 'Odometry Only'
    if data.get('odo_model_params') is not None:
        _, _, _, odo_config = data['odo_model_params']
        odo_label = f"Odometry Only (NARX {odo_config['model_type']})"
    
    ax_main.plot(odom_traj[:, 0], odom_traj[:, 1], 'b--', linewidth=1.5,
                label=odo_label, alpha=0.6)
    
    for name, traj in trajectories.items():
        ax_main.plot(traj[:, 0], traj[:, 1], 
                    color=colors[name], 
                    linestyle=linestyles[name],
                    linewidth=linewidths[name],
                    label=f'{name} Estimate',
                    alpha=0.8)
    
    # Mark start and end
    ax_main.plot(data['initial_pose'][0], data['initial_pose'][1], 
                'go', markersize=12, label='Start', zorder=10)
    if ref_traj is not None:
        ax_main.plot(ref_traj[-1, 0], ref_traj[-1, 1], 
                    'r*', markersize=15, label='End', zorder=10)
    
    ax_main.set_xlabel('X Position (m)', fontsize=12)
    ax_main.set_ylabel('Y Position (m)', fontsize=12)
    ax_main.set_title('Filter Comparison: Trajectory Estimates', fontsize=14, fontweight='bold')
    ax_main.set_xlim(map_info["xlimits"])
    ax_main.set_ylim(map_info["ylimits"])
    ax_main.legend(loc='best', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    if ref_traj is not None:
        # Position error over time
        ax_error = fig.add_subplot(gs[0, 2])
        n_steps = min([len(traj) for traj in trajectories.values()] + [len(ref_traj)])
        time_steps = np.arange(n_steps)
        
        for name, traj in trajectories.items():
            pos_error = traj[:n_steps, :2] - ref_traj[:n_steps, :2]
            pos_distances = np.linalg.norm(pos_error, axis=1)
            ax_error.plot(time_steps, pos_distances, 
                         color=colors[name], linewidth=1.5, 
                         label=name, alpha=0.8)
        
        ax_error.set_xlabel('Time Step', fontsize=11)
        ax_error.set_ylabel('Position Error (m)', fontsize=11)
        ax_error.set_title('Position Error Over Time', fontsize=12, fontweight='bold')
        ax_error.legend(loc='best', fontsize=10)
        ax_error.grid(True, alpha=0.3)
        
        # Error distribution (box plot)
        ax_box = fig.add_subplot(gs[1, 2])
        error_data = []
        labels = []
        for name, traj in trajectories.items():
            pos_error = traj[:n_steps, :2] - ref_traj[:n_steps, :2]
            pos_distances = np.linalg.norm(pos_error, axis=1)
            error_data.append(pos_distances)
            labels.append(name)
        
        bp = ax_box.boxplot(error_data, label=labels, patch_artist=True)
        for patch, name in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.6)
        
        ax_box.set_ylabel('Position Error (m)', fontsize=11)
        ax_box.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax_box.grid(True, alpha=0.3, axis='y')
    else:
        # If no reference, show execution time comparison
        ax_time = fig.add_subplot(gs[0, 2])
        names = list(exec_times.keys())
        times = [exec_times[name] for name in names]
        bars = ax_time.bar(names, times, color=[colors[name] for name in names], alpha=0.7)
        ax_time.set_ylabel('Execution Time (s)', fontsize=11)
        ax_time.set_title('Computational Performance', fontsize=12, fontweight='bold')
        ax_time.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_time.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # Add info text
    info_text = f"Data: {len(data['odo_deltas'])} steps\n"
    is_narx = data['laser_model_params'][0]
    if is_narx:
        _, _, _, narx_config = data['laser_model_params']
        info_text += f"Laser Model: NARX ({narx_config['model_type']}, n_lags={narx_config['n_lags']})\n"
    else:
        info_text += "Laser Model: Linear\n"
    
    # Add odometry model info
    if data.get('odo_model_params') is not None:
        _, _, _, odo_config = data['odo_model_params']
        info_text += f"Odo Model: NARX ({odo_config['model_type']}, n_lags={odo_config['n_lags']})\n"
    else:
        info_text += "Odo Model: Additive\n"
    
    info_text += f"Reference: {'Available' if ref_traj is not None else 'Not available'}"
    
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    
    return fig


def save_comparison_results(all_metrics: list, exec_times: dict, output_json: str):
    """Save numerical comparison results to JSON file."""
    results = {
        'metrics': all_metrics,
        'execution_times': exec_times,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Comparison results saved to: {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare UKF, EKF, and Particle Filter performance"
    )
    parser.add_argument(
        "--odo-diff",
        default="odo_dec_diff_trimmed.csv",
        help="Path to odometry differences CSV",
    )
    parser.add_argument(
        "--laser", 
        default="laser_dec_trimmed.csv", 
        help="Path to laser data CSV"
    )
    parser.add_argument(
        "--model", 
        default="laser_model.json", 
        help="Path to laser model JSON"
    )
    parser.add_argument(
        "--odo-model",
        default=None,
        help="Path to odometry NARX model JSON (optional)"
    )
    parser.add_argument(
        "--map-info", 
        default="map_info.json", 
        help="Path to map info JSON"
    )
    parser.add_argument(
        "--reference",
        default="ref_dec_trimmed.csv",
        help="Path to reference trajectory CSV (ground truth)"
    )
    parser.add_argument(
        "--ukf-params",
        default="ukf_tuned.json",
        help="Path to UKF tuned parameters (optional)"
    )
    parser.add_argument(
        "--ekf-params",
        default="ekf_tuned.json",
        help="Path to EKF tuned parameters (optional)"
    )
    parser.add_argument(
        "--pf-params",
        default="pf_tuned.json",
        help="Path to PF tuned parameters (optional)"
    )
    parser.add_argument(
        "--q-std",
        type=float,
        nargs=3,
        default=[0.02, 0.02, 0.05],
        help="Default process noise std (if no tuned params)"
    )
    parser.add_argument(
        "--r-std",
        type=float,
        nargs=3,
        default=[0.05, 0.05, 0.1],
        help="Default measurement noise std (if no tuned params)"
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=1000,
        help="Number of particles for PF (if no tuned params)"
    )
    parser.add_argument(
        "--output-plot",
        default="filter_comparison.png",
        help="Output path for comparison plot"
    )
    parser.add_argument(
        "--output-json",
        default="filter_comparison.json",
        help="Output path for numerical results"
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show plot interactively after saving"
    )
    parser.add_argument(
        "--skip-statistical-tests",
        action="store_true",
        help="Skip statistical significance tests"
    )
    return parser.parse_args()


def load_filter_params(param_file: str, defaults: dict) -> dict:
    """Load filter parameters from JSON file or use defaults."""
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        print(f"  Loaded parameters from {param_file}")
        return params
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"  Using default parameters (file not found: {param_file})")
        return defaults


def main():
    args = parse_args()
    
    print("="*80)
    print("FILTER COMPARISON TOOL")
    print("="*80)
    print(f"Comparing: UKF, EKF, and Particle Filter")
    print(f"Model: {args.model}")
    print(f"Reference: {args.reference}")
    
    # Load data
    print("\nLoading data...")
    data = load_data(args)
    
    # Check laser model type
    is_narx = data['laser_model_params'][0]
    if is_narx:
        _, _, _, narx_config = data['laser_model_params']
        print(f"Using laser NARX model: {narx_config['model_type']}, n_lags={narx_config['n_lags']}")
    else:
        print("Using linear laser model")
    
    # Check odometry model type
    if data.get('odo_model_params') is not None:
        _, _, _, odo_config = data['odo_model_params']
        print(f"Using odometry NARX model: {odo_config['model_type']}, n_lags={odo_config['n_lags']}")
    else:
        print("Using simple additive odometry model")
    
    # Load or use default parameters for each filter
    print("\nLoading filter parameters...")
    default_params = {
        'q_std': tuple(args.q_std),
        'r_std': tuple(args.r_std),
    }
    
    ukf_params = load_filter_params(args.ukf_params, default_params)
    ekf_params = load_filter_params(args.ekf_params, default_params)
    
    pf_defaults = {
        **default_params,
        'n_particles': args.n_particles,
        'resample_threshold': 0.5,
    }
    pf_params = load_filter_params(args.pf_params, pf_defaults)
    
    # Run all filters
    trajectories = {}
    exec_times = {}
    
    traj_ukf, time_ukf, _ = run_filter(UKFEstimator, "UKF", data, ukf_params)
    trajectories['UKF'] = traj_ukf
    exec_times['UKF'] = time_ukf
    
    traj_ekf, time_ekf, _ = run_filter(EKFEstimator, "EKF", data, ekf_params)
    trajectories['EKF'] = traj_ekf
    exec_times['EKF'] = time_ekf
    
    traj_pf, time_pf, pf_stats = run_filter(ParticleFilter, "PF", data, pf_params)
    trajectories['PF'] = traj_pf
    exec_times['PF'] = time_pf
    
    # Print execution times
    print("\n" + "="*80)
    print("EXECUTION TIME COMPARISON")
    print("="*80)
    for name, exec_time in exec_times.items():
        speedup = exec_times['PF'] / exec_time if exec_time > 0 else 0
        print(f"{name:>6}: {exec_time:8.4f} seconds (speedup vs PF: {speedup:5.2f}x)")
    print("="*80)
    
    # Compute metrics if reference is available
    if data['ref_traj'] is not None:
        all_metrics = []
        for name, traj in trajectories.items():
            metrics = compute_metrics(traj, data['ref_traj'], name)
            all_metrics.append(metrics)
        
        # Print metrics table
        print_metrics_table(all_metrics)
        
        # Statistical tests
        if not args.skip_statistical_tests:
            statistical_tests(trajectories, data['ref_traj'])
        
        # Save numerical results
        save_comparison_results(all_metrics, exec_times, args.output_json)
    else:
        print("\nWarning: No reference trajectory provided, skipping error metrics")
        all_metrics = None
    
    # Create and save comparison plot
    print("\nGenerating comparison plot...")
    fig = plot_comparison(data, trajectories, exec_times, args.output_plot)
    
    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
