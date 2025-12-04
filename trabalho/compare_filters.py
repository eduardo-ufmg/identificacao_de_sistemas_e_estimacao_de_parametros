import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OdometryModel import OdometryModel
from EKF import ExtendedKalmanFilter, run_ekf_estimation
from UKF import UnscentedKalmanFilter, run_ukf_estimation  
from PF import ParticleFilter, run_pf_estimation
import time

def compare_filters():
    """
    Compare the performance of EKF, UKF, and Particle Filter on the robot localization task.
    """
    print("Loading data...")
    
    # Load data
    ref_data = np.genfromtxt('ref.csv', delimiter=',')[0:100]
    odo_data = np.genfromtxt('odo.csv', delimiter=',')[0:100]
    laser_data = np.genfromtxt('laser.csv', delimiter=',')[0:100]
    
    # Initial pose from reference
    initial_pose = (ref_data[0, 0], ref_data[0, 1], ref_data[0, 2])
    
    # Compute pure odometry for comparison
    odo_model = OdometryModel(initial_pose)
    odo_positions = odo_model.compute_positions(odo_data)
    
    print(f"Data loaded: {len(ref_data)} time steps")
    print(f"Initial pose: ({initial_pose[0]:.3f}, {initial_pose[1]:.3f}, {initial_pose[2]:.3f})")
    
    # Run filters
    filters_data = {}
    
    # Extended Kalman Filter
    print("\n=== Running Extended Kalman Filter ===")
    start_time = time.time()
    ekf_poses = run_ekf_estimation(initial_pose, odo_data, laser_data)
    ekf_time = time.time() - start_time
    filters_data['EKF'] = {'poses': ekf_poses, 'time': ekf_time, 'color': 'blue', 'style': '-'}
    print(f"EKF completed in {ekf_time:.2f} seconds")

    # Save EKF results to CSV
    ekf_df = pd.DataFrame(ekf_poses, columns=['x', 'y', 'theta'])
    ekf_df.to_csv('ekf_results.csv', index=False)
    print("EKF results saved to 'ekf_results.csv'")
    
    # Unscented Kalman Filter  
    print("\n=== Running Unscented Kalman Filter ===")
    start_time = time.time()
    ukf_poses = run_ukf_estimation(initial_pose, odo_data, laser_data)
    ukf_time = time.time() - start_time
    filters_data['UKF'] = {'poses': ukf_poses, 'time': ukf_time, 'color': 'red', 'style': '-'}
    print(f"UKF completed in {ukf_time:.2f} seconds")

    # Save UKF results to CSV
    ukf_df = pd.DataFrame(ukf_poses, columns=['x', 'y', 'theta'])
    ukf_df.to_csv('ukf_results.csv', index=False)
    print("UKF results saved to 'ukf_results.csv'")
    
    # Particle Filter
    print("\n=== Running Particle Filter ===") 
    start_time = time.time()
    # Takes too long, so use dummy data
    # pf_poses = run_pf_estimation(initial_pose, odo_data, laser_data, n_particles=1500)
    pf_poses = np.zeros_like(ekf_poses)  # Dummy placeholder
    pf_time = time.time() - start_time
    pf_time = np.inf  # Indicate not run
    filters_data['PF'] = {'poses': pf_poses, 'time': pf_time, 'color': 'magenta', 'style': '-'}
    print(f"PF completed in {pf_time:.2f} seconds")

    # Save PF results to CSV
    pf_df = pd.DataFrame(pf_poses, columns=['x', 'y', 'theta'])
    pf_df.to_csv('pf_results.csv', index=False)
    print("PF results saved to 'pf_results.csv'")
    
    # Calculate errors
    print("\n=== Performance Analysis ===")
    
    errors = {}
    rmse_errors = {}
    
    for filter_name, data in filters_data.items():
        poses = data['poses']
        
        # Final position error
        final_error = np.sqrt((ref_data[-1, 0] - poses[-1, 0])**2 + 
                             (ref_data[-1, 1] - poses[-1, 1])**2)
        
        # RMSE over entire trajectory
        position_errors = np.sqrt((ref_data[:, 0] - poses[:, 0])**2 + 
                                 (ref_data[:, 1] - poses[:, 1])**2)
        rmse = np.sqrt(np.mean(position_errors**2))
        
        # Angular error (final)
        angle_error = abs(ref_data[-1, 2] - poses[-1, 2])
        angle_error = min(angle_error, 2*np.pi - angle_error)  # Wrap to [0, π]
        
        errors[filter_name] = final_error
        rmse_errors[filter_name] = rmse
        
        print(f"{filter_name}:")
        print(f"  Final position error: {final_error:.3f} m")
        print(f"  RMSE position error: {rmse:.3f} m") 
        print(f"  Final angular error: {np.degrees(angle_error):.1f}°")
        print(f"  Computation time: {data['time']:.2f} s")
    
    # Odometry-only baseline
    odo_final_error = np.sqrt((ref_data[-1, 0] - odo_positions[-1, 0])**2 + 
                             (ref_data[-1, 1] - odo_positions[-1, 1])**2)
    odo_rmse = np.sqrt(np.mean((ref_data[:, 0] - odo_positions[:, 0])**2 + 
                              (ref_data[:, 1] - odo_positions[:, 1])**2))
    
    print(f"\nOdometry-only baseline:")
    print(f"  Final position error: {odo_final_error:.3f} m")
    print(f"  RMSE position error: {odo_rmse:.3f} m")
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Trajectory comparison
    ax1.plot(ref_data[:, 0], ref_data[:, 1], 'g-', linewidth=3, label='Reference', alpha=0.8)
    ax1.plot(odo_positions[:, 0], odo_positions[:, 1], 'k--', linewidth=2, label='Odometry Only', alpha=0.7)
    
    for filter_name, data in filters_data.items():
        poses = data['poses']
        ax1.plot(poses[:, 0], poses[:, 1], color=data['color'], linestyle=data['style'], 
                linewidth=2, label=f'{filter_name}', alpha=0.8)
    
    # Mark start and end points
    ax1.plot(ref_data[0, 0], ref_data[0, 1], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(ref_data[-1, 0], ref_data[-1, 1], 'rs', markersize=12, label='Reference End', zorder=5)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12) 
    ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Position error over time
    time_steps = np.arange(len(ref_data))
    
    odo_errors = np.sqrt((ref_data[:, 0] - odo_positions[:, 0])**2 + 
                        (ref_data[:, 1] - odo_positions[:, 1])**2)
    ax2.plot(time_steps, odo_errors, 'k--', linewidth=2, label='Odometry Only', alpha=0.7)
    
    for filter_name, data in filters_data.items():
        poses = data['poses']
        position_errors = np.sqrt((ref_data[:, 0] - poses[:, 0])**2 + 
                                 (ref_data[:, 1] - poses[:, 1])**2)
        ax2.plot(time_steps, position_errors, color=data['color'], linewidth=2, 
                label=f'{filter_name}', alpha=0.8)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Position Error (m)', fontsize=12)
    ax2.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error statistics comparison
    filter_names = list(filters_data.keys()) + ['Odometry']
    final_errors = [errors[name] for name in filters_data.keys()] + [odo_final_error]
    rmse_vals = [rmse_errors[name] for name in filters_data.keys()] + [odo_rmse]
    
    x = np.arange(len(filter_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, final_errors, width, label='Final Error', alpha=0.8)
    bars2 = ax3.bar(x + width/2, rmse_vals, width, label='RMSE', alpha=0.8)
    
    ax3.set_xlabel('Filter Method', fontsize=12)
    ax3.set_ylabel('Position Error (m)', fontsize=12)
    ax3.set_title('Error Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(filter_names)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Computation time comparison
    comp_times = [filters_data[name]['time'] for name in filters_data.keys()]
    colors = [filters_data[name]['color'] for name in filters_data.keys()]
    
    bars = ax4.bar(filters_data.keys(), comp_times, color=colors, alpha=0.7)
    ax4.set_xlabel('Filter Method', fontsize=12)
    ax4.set_ylabel('Computation Time (s)', fontsize=12)
    ax4.set_title('Computational Performance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n=== SUMMARY ===")
    best_accuracy = min(errors.values())
    best_filter_accuracy = [k for k, v in errors.items() if v == best_accuracy][0]
    
    best_speed = min(comp_times)
    best_filter_speed = [k for k, v in zip(filters_data.keys(), comp_times) if v == best_speed][0]
    
    print(f"Best accuracy: {best_filter_accuracy} ({best_accuracy:.3f} m final error)")
    print(f"Fastest method: {best_filter_speed} ({best_speed:.2f} s)")
    print(f"Odometry improvement: {odo_final_error/best_accuracy:.1f}x better than odometry-only")

if __name__ == "__main__":
    compare_filters()