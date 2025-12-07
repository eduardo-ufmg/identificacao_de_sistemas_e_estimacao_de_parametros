"""
Compare all localization filters (PF, EKF, UKF) across multiple decimation levels.

This script iterates through decimated datasets from smallest to largest,
performing complete filter runs and comparisons at each decimation level,
saving all results to disk before proceeding to the next level.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
import os
from pathlib import Path
from PF import ParticleFilter
from EKF import ExtendedKalmanFilter
from UKF import UnscentedKalmanFilter
from OdometryDynamicModel import OdometryDynamicModel
import time


# Define decimation levels to process (in order from smallest to largest)
DECIMATION_LEVELS = [
    {
        'name': 'aggressive',
        'odo_file': 'odo_diff_aggressive.csv',
        'ref_file': 'ref_aggressive.csv',
        'laser_file': None,  # Use standard laser if None
        'description': 'Aggressive decimation (coarse sensor readings)'
    },
    {
        'name': 'balanced',
        'odo_file': 'odo_diff_balanced.csv',
        'ref_file': 'ref_balanced.csv',
        'laser_file': None,
        'description': 'Balanced decimation (medium sampling)'
    },
    {
        'name': 'conservative',
        'odo_file': 'odo_diff_conservative.csv',
        'ref_file': 'ref_conservative.csv',
        'laser_file': 'laser_conservative.csv',
        'description': 'Conservative decimation (fine sampling)'
    },
    {
        'name': 'full',
        'odo_file': 'odo_diff.csv',
        'ref_file': 'ref.csv',
        'laser_file': 'laser.csv',
        'description': 'No decimation (full dataset)'
    }
]


def verify_files_exist(config):
    """Check if required files exist."""
    odo_file = config['odo_file']
    ref_file = config['ref_file']
    laser_file = config['laser_file'] if config['laser_file'] else 'laser.csv'
    
    missing = []
    for f in [odo_file, ref_file, laser_file]:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print(f"  WARNING: Missing files for {config['name']}: {missing}")
        return False
    return True


def run_all_filters(config, tuned_params=None):
    """
    Run all three filters with specified dataset.
    
    Args:
        config: Decimation configuration dict
        tuned_params: Optional dict with tuned parameters for each filter
        
    Returns:
        Dictionary with filter results
    """
    odo_file = config['odo_file']
    ref_file = config['ref_file']
    laser_file = config['laser_file'] if config['laser_file'] else 'laser.csv'
    
    # Load initial pose
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info['initial_pose'])
    
    results = {}
    
    # Prepare filter configs
    pf_config = None
    ekf_config = None
    ukf_config = None
    
    if tuned_params:
        if 'PF' in tuned_params:
            pf_config = tuned_params['PF'].get('best_params', {})
        if 'EKF' in tuned_params:
            ekf_config = tuned_params['EKF'].get('best_params', {})
        if 'UKF' in tuned_params:
            ukf_config = tuned_params['UKF'].get('best_params', {})
    
    # Run Particle Filter
    print(f"  Running Particle Filter...")
    start_time = time.time()
    try:
        pf = ParticleFilter(num_particles=500, config=pf_config)
        pf_estimates = pf.run_filter(
            odo_diff_file=odo_file,
            laser_file=laser_file,
            initial_pose=initial_pose,
            initial_uncertainty=np.array([0.5, 0.5, 0.1])
        )
        pf_time = time.time() - start_time
        results['PF'] = {'estimates': pf_estimates, 'time': pf_time}
    except Exception as e:
        print(f"    ERROR in Particle Filter: {e}")
        results['PF'] = {'error': str(e)}
    
    # Run Extended Kalman Filter
    print(f"  Running Extended Kalman Filter...")
    start_time = time.time()
    try:
        ekf = ExtendedKalmanFilter(config=ekf_config)
        ekf_estimates, ekf_cov = ekf.run_filter(
            odo_diff_file=odo_file,
            laser_file=laser_file,
            initial_state=initial_pose,
            initial_covariance=np.diag([0.1, 0.1, 0.05])**2
        )
        ekf_time = time.time() - start_time
        results['EKF'] = {'estimates': ekf_estimates, 'covariances': ekf_cov, 'time': ekf_time}
    except Exception as e:
        print(f"    ERROR in Extended Kalman Filter: {e}")
        results['EKF'] = {'error': str(e)}
    
    # Run Unscented Kalman Filter
    print(f"  Running Unscented Kalman Filter...")
    start_time = time.time()
    try:
        ukf = UnscentedKalmanFilter(config=ukf_config)
        ukf_estimates, ukf_cov = ukf.run_filter(
            odo_diff_file=odo_file,
            laser_file=laser_file,
            initial_state=initial_pose,
            initial_covariance=np.diag([0.1, 0.1, 0.05])**2
        )
        ukf_time = time.time() - start_time
        results['UKF'] = {'estimates': ukf_estimates, 'covariances': ukf_cov, 'time': ukf_time}
    except Exception as e:
        print(f"    ERROR in Unscented Kalman Filter: {e}")
        results['UKF'] = {'error': str(e)}
    
    # Compute dynamic odometry for comparison
    print(f"  Computing Dynamic Odometry baseline...")
    try:
        odo_dyn_model = OdometryDynamicModel(initial_pose=tuple(initial_pose))
        odo_diff_data = OdometryDynamicModel.load_from_csv(odo_file)
        odo_estimates = odo_dyn_model.compute_positions(odo_diff_data)
        results['Odometry'] = {'estimates': odo_estimates}
    except Exception as e:
        print(f"    ERROR in Odometry: {e}")
        results['Odometry'] = {'error': str(e)}
    
    return results


def compute_statistics(estimates, reference):
    """Compute error statistics."""
    if estimates is None or reference is None:
        return None
    
    if len(estimates) != len(reference):
        print(f"    WARNING: Estimate length {len(estimates)} != reference length {len(reference)}")
        # Trim to shorter length
        min_len = min(len(estimates), len(reference))
        estimates = estimates[:min_len]
        reference = reference[:min_len]
    
    position_errors = np.sqrt((reference[:, 0] - estimates[:, 0])**2 + 
                             (reference[:, 1] - estimates[:, 1])**2)
    
    angle_errors = np.abs(np.arctan2(np.sin(reference[:, 2] - estimates[:, 2]),
                                     np.cos(reference[:, 2] - estimates[:, 2])))
    
    return {
        'final_error': position_errors[-1],
        'mean_error': np.mean(position_errors),
        'max_error': np.max(position_errors),
        'std_error': np.std(position_errors),
        'final_angle_error': np.degrees(angle_errors[-1]),
        'mean_angle_error': np.degrees(np.mean(angle_errors))
    }


def save_results(config, results, reference):
    """Save filter results and statistics to files."""
    prefix = f"results_{config['name']}"
    
    # Save estimates
    for filter_name, data in results.items():
        if 'error' not in data and 'estimates' in data:
            filename = f"{prefix}_{filter_name}.csv"
            np.savetxt(filename, data['estimates'], delimiter=',')
            print(f"    Saved {filename}")
    
    # Save statistics
    stats_file = f"{prefix}_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Filter Comparison Results for {config['description']}\n")
        f.write(f"Dataset: {config['name']}\n")
        f.write(f"=" * 70 + "\n\n")
        
        for filter_name, data in results.items():
            f.write(f"{filter_name}:\n")
            
            if 'error' in data:
                f.write(f"  Status: ERROR\n")
                f.write(f"  Details: {data['error']}\n\n")
                continue
            
            if 'estimates' not in data:
                f.write(f"  Status: No estimates\n\n")
                continue
            
            stats = compute_statistics(data['estimates'], reference)
            if stats:
                f.write(f"  Final Position Error:    {stats['final_error']:.4f} m\n")
                f.write(f"  Mean Position Error:     {stats['mean_error']:.4f} m\n")
                f.write(f"  Max Position Error:      {stats['max_error']:.4f} m\n")
                f.write(f"  Std Position Error:      {stats['std_error']:.4f} m\n")
                f.write(f"  Final Angle Error:       {stats['final_angle_error']:.2f}°\n")
                f.write(f"  Mean Angle Error:        {stats['mean_angle_error']:.2f}°\n")
            
            if 'time' in data:
                f.write(f"  Computation Time:        {data['time']:.2f} seconds\n")
            
            f.write("\n")
    
    print(f"    Saved {stats_file}")
    return stats_file


def plot_comparison(config, results, reference):
    """
    Create comprehensive comparison plots for a decimation level.
    
    Args:
        config: Decimation configuration
        results: Dictionary with filter results
        reference: Reference trajectory
    """
    # Load map
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    
    map_image = imread('map.png')
    xlimits = map_info['xlimits']
    ylimits = map_info['ylimits']
    map_extent = (xlimits[0], xlimits[1], ylimits[0], ylimits[1])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Filter Comparison: {config['description']}", fontsize=16, fontweight='bold')
    
    # Main trajectory plot
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(map_image, extent=map_extent, alpha=0.6, cmap='gray')
    ax1.plot(reference[:, 0], reference[:, 1], 'g-', linewidth=3, label='Reference', alpha=0.9)
    
    colors = {'PF': 'blue', 'EKF': 'red', 'UKF': 'purple', 'Odometry': 'orange'}
    linestyles = {'PF': '-', 'EKF': '--', 'UKF': '-.', 'Odometry': ':'}
    
    for name, data in results.items():
        if 'estimates' in data:
            est = data['estimates']
            ax1.plot(est[:, 0], est[:, 1], color=colors[name], 
                    linestyle=linestyles[name], linewidth=2, label=name, alpha=0.8)
    
    ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Position error over time
    ax2 = plt.subplot(2, 2, 2)
    for name, data in results.items():
        if 'estimates' in data:
            est = data['estimates']
            if len(est) == len(reference):
                errors = np.sqrt((reference[:, 0] - est[:, 0])**2 + 
                                (reference[:, 1] - est[:, 1])**2)
                ax2.plot(errors, color=colors[name], linestyle=linestyles[name], 
                        linewidth=2, label=name, alpha=0.8)
    
    ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Angle error over time
    ax3 = plt.subplot(2, 2, 3)
    for name, data in results.items():
        if 'estimates' in data:
            est = data['estimates']
            if len(est) == len(reference):
                angle_errors = np.abs(np.arctan2(np.sin(reference[:, 2] - est[:, 2]),
                                                np.cos(reference[:, 2] - est[:, 2])))
                ax3.plot(np.degrees(angle_errors), color=colors[name], 
                        linestyle=linestyles[name], linewidth=2, label=name, alpha=0.8)
    
    ax3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Angle Error (degrees)', fontsize=12, fontweight='bold')
    ax3.set_title('Orientation Error Over Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Error statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"Performance Summary\n{config['description']}\n" + "="*50 + "\n\n"
    
    for name, data in results.items():
        if 'estimates' in data:
            stats = compute_statistics(data['estimates'], reference)
            if stats:
                stats_text += f"{name}:\n"
                stats_text += f"  Final: {stats['final_error']:.3f}m\n"
                stats_text += f"  Mean:  {stats['mean_error']:.3f}m\n"
                stats_text += f"  Max:   {stats['max_error']:.3f}m\n"
                if 'time' in data:
                    stats_text += f"  Time:  {data['time']:.2f}s\n"
                stats_text += "\n"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    filename = f"comparison_{config['name']}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    Saved {filename}")
    plt.close(fig)


def process_decimation_level(config, tuned_params=None):
    """
    Process a single decimation level: run filters, compute stats, save results, create plots.
    
    Args:
        config: Decimation configuration dict
        tuned_params: Optional dict with tuned parameters for each filter
        
    Returns:
        Boolean indicating success
    """
    print(f"\n{'='*70}")
    print(f"Processing: {config['description']}")
    print(f"Dataset: {config['name']}")
    if tuned_params:
        print(f"Using TUNED parameters")
    print(f"{'='*70}")
    
    # Verify files exist
    if not verify_files_exist(config):
        print(f"SKIPPING {config['name']} - missing files")
        return False
    
    # Load reference trajectory
    print(f"Loading reference trajectory from {config['ref_file']}...")
    reference = np.genfromtxt(config['ref_file'], delimiter=',')
    print(f"  Loaded {len(reference)} reference points")
    
    # Run all filters
    print(f"Running filters...")
    results = run_all_filters(config, tuned_params=tuned_params)
    
    # Save results and statistics
    print(f"Saving results...")
    save_results(config, results, reference)
    
    # Create plots
    print(f"Creating plots...")
    plot_comparison(config, results, reference)
    
    # Print summary to console
    print(f"Summary for {config['name']}:")
    for name, data in results.items():
        if 'estimates' in data:
            stats = compute_statistics(data['estimates'], reference)
            if stats:
                print(f"  {name}: Final={stats['final_error']:.3f}m, Mean={stats['mean_error']:.3f}m, Max={stats['max_error']:.3f}m", end="")
                if 'time' in data:
                    print(f", Time={data['time']:.1f}s")
                else:
                    print()
    
    return True


def main(tuned_params=None):
    """Main execution: iterate through decimation levels and process each.
    
    Args:
        tuned_params: Optional dict with tuned parameters for each filter
    """
    print("\n" + "="*70)
    print("MULTI-LEVEL FILTER COMPARISON ANALYSIS")
    print("="*70)
    print("\nThis script will process each decimation level from smallest to largest,")
    print("running complete filter cycles and saving all results before proceeding.\n")
    
    if tuned_params:
        print("Using TUNED parameters from tune_filters.py\n")
    
    start_time_total = time.time()
    processed_count = 0
    
    # Process each decimation level
    for i, config in enumerate(DECIMATION_LEVELS):
        try:
            success = process_decimation_level(config, tuned_params=tuned_params)
            if success:
                processed_count += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nERROR processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time_total
    
    # Final summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed {processed_count} decimation levels")
    print(f"Total computation time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nOutput files saved with pattern: results_<level>_<filter>.csv")
    print(f"Plots saved with pattern: comparison_<level>.png")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
