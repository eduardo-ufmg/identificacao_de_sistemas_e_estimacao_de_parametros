"""
Compare all localization filters (PF, EKF, UKF) against reference and odometry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
from PF import ParticleFilter
from EKF import ExtendedKalmanFilter
from UKF import UnscentedKalmanFilter
from OdometryDynamicModel import OdometryDynamicModel
import time


def run_all_filters():
    """Run all three filters and save results."""
    
    # Load initial pose
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info['initial_pose'])
    
    results = {}
    
    # Run Particle Filter
    print("\n" + "="*60)
    print("PARTICLE FILTER")
    print("="*60)
    start_time = time.time()
    pf = ParticleFilter(num_particles=500)
    pf_estimates = pf.run_filter(initial_pose=initial_pose,
                                  initial_uncertainty=np.array([0.5, 0.5, 0.1]))
    pf_time = time.time() - start_time
    np.savetxt('pf_estimates.csv', pf_estimates, delimiter=',')
    results['PF'] = {'estimates': pf_estimates, 'time': pf_time}
    
    # Run Extended Kalman Filter
    print("\n" + "="*60)
    print("EXTENDED KALMAN FILTER")
    print("="*60)
    start_time = time.time()
    ekf = ExtendedKalmanFilter()
    ekf_estimates, ekf_cov = ekf.run_filter(initial_state=initial_pose,
                                            initial_covariance=np.diag([0.1, 0.1, 0.05])**2)
    ekf_time = time.time() - start_time
    np.savetxt('ekf_estimates.csv', ekf_estimates, delimiter=',')
    results['EKF'] = {'estimates': ekf_estimates, 'covariances': ekf_cov, 'time': ekf_time}
    
    # Run Unscented Kalman Filter
    print("\n" + "="*60)
    print("UNSCENTED KALMAN FILTER")
    print("="*60)
    start_time = time.time()
    ukf = UnscentedKalmanFilter()
    ukf_estimates, ukf_cov = ukf.run_filter(initial_state=initial_pose,
                                            initial_covariance=np.diag([0.1, 0.1, 0.05])**2)
    ukf_time = time.time() - start_time
    np.savetxt('ukf_estimates.csv', ukf_estimates, delimiter=',')
    results['UKF'] = {'estimates': ukf_estimates, 'covariances': ukf_cov, 'time': ukf_time}
    
    # Also compute dynamic odometry for comparison
    print("\n" + "="*60)
    print("DYNAMIC ODOMETRY (baseline)")
    print("="*60)
    odo_dyn_model = OdometryDynamicModel(initial_pose=tuple(initial_pose))
    odo_diff_data = OdometryDynamicModel.load_from_csv('odo_diff.csv')
    odo_estimates = odo_dyn_model.compute_positions(odo_diff_data)
    results['Odometry'] = {'estimates': odo_estimates}
    
    return results


def compute_statistics(estimates, reference):
    """Compute error statistics."""
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


def plot_comparison(results, reference):
    """Create comprehensive comparison plots."""
    
    # Load map
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    
    map_image = imread('map.png')
    xlimits = map_info['xlimits']
    ylimits = map_info['ylimits']
    map_extent = (xlimits[0], xlimits[1], ylimits[0], ylimits[1])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Main trajectory plot
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(map_image, extent=map_extent, alpha=0.6, cmap='gray')
    ax1.plot(reference[:, 0], reference[:, 1], 'g-', linewidth=3, label='Reference', alpha=0.9)
    
    colors = {'PF': 'blue', 'EKF': 'red', 'UKF': 'purple', 'Odometry': 'orange'}
    linestyles = {'PF': '-', 'EKF': '--', 'UKF': '-.', 'Odometry': ':'}
    
    for name, data in results.items():
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
        est = data['estimates']
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
        est = data['estimates']
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
    
    stats_text = "Performance Comparison\n" + "="*50 + "\n\n"
    
    for name, data in results.items():
        stats = compute_statistics(data['estimates'], reference)
        stats_text += f"{name}:\n"
        stats_text += f"  Final Error:      {stats['final_error']:.3f} m\n"
        stats_text += f"  Mean Error:       {stats['mean_error']:.3f} m\n"
        stats_text += f"  Max Error:        {stats['max_error']:.3f} m\n"
        stats_text += f"  Std Error:        {stats['std_error']:.3f} m\n"
        stats_text += f"  Final Angle Err:  {stats['final_angle_error']:.2f}°\n"
        stats_text += f"  Mean Angle Err:   {stats['mean_angle_error']:.2f}°\n"
        if 'time' in data:
            stats_text += f"  Computation Time: {data['time']:.2f} s\n"
        stats_text += "\n"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to 'filter_comparison.png'")
    
    return fig


def print_summary(results, reference):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for name, data in results.items():
        stats = compute_statistics(data['estimates'], reference)
        print(f"\n{name}:")
        print(f"  Final Position Error:    {stats['final_error']:.4f} m")
        print(f"  Mean Position Error:     {stats['mean_error']:.4f} m")
        print(f"  Max Position Error:      {stats['max_error']:.4f} m")
        print(f"  Std Position Error:      {stats['std_error']:.4f} m")
        print(f"  Final Angle Error:       {stats['final_angle_error']:.2f}°")
        print(f"  Mean Angle Error:        {stats['mean_angle_error']:.2f}°")
        if 'time' in data:
            print(f"  Computation Time:        {data['time']:.2f} seconds")


if __name__ == '__main__':
    print("Starting filter comparison...")
    print("This may take several minutes depending on your system.")
    
    # Load reference trajectory
    reference = np.genfromtxt('ref.csv', delimiter=',')
    
    # Run all filters
    results = run_all_filters()
    
    # Print summary
    print_summary(results, reference)
    
    # Create plots
    plot_comparison(results, reference)
    
    plt.show()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Results saved to:")
    print("  - pf_estimates.csv")
    print("  - ekf_estimates.csv")
    print("  - ukf_estimates.csv")
    print("  - filter_comparison.png")
    print("="*60)
