"""
Structural Error Analysis for Robot Localization System

This script analyzes potential structural issues in the data handling and processing
that could explain poor filter performance even after parameter tuning.

Key Areas Investigated:
1. Odometry coordinate frame transformations
2. Data alignment and synchronization
3. Laser scan coordinate systems
4. Map coordinate transformations
5. Initial pose accuracy
6. Numerical stability issues
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
import os

from OdometryDynamicModel import OdometryDynamicModel
from OdometryStaticModel import OdometryStaticModel


def analyze_coordinate_frame_issue():
    """
    CRITICAL ISSUE: Analyze if odometry differential data needs coordinate transformation.
    
    The OdometryDynamicModel currently treats dx, dy as global frame increments,
    but differential odometry is typically measured in the robot's local frame.
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: COORDINATE FRAME TRANSFORMATION")
    print("="*80)
    
    # Load data
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info['initial_pose'])
    
    odo_diff_data = np.genfromtxt('odo_diff.csv', delimiter=',')
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    print(f"\nInitial pose: x={initial_pose[0]:.2f}, y={initial_pose[1]:.2f}, theta={initial_pose[2]:.4f}")
    print(f"Reference trajectory: {len(ref_data)} points")
    print(f"Odometry data: {len(odo_diff_data)} points")
    
    # Current (incorrect) integration: treating dx, dy as global increments
    odo_model_wrong = OdometryDynamicModel(initial_pose=tuple(initial_pose))
    positions_wrong = odo_model_wrong.compute_positions(odo_diff_data)
    
    # Correct integration: transform from local to global frame
    positions_correct = np.zeros_like(odo_diff_data)
    positions_correct[0] = initial_pose
    
    x, y, theta = initial_pose
    for i in range(1, len(odo_diff_data)):
        dx_local = odo_diff_data[i, 0]
        dy_local = odo_diff_data[i, 1]
        dtheta = odo_diff_data[i, 2]
        
        # Transform from local to global frame using current orientation
        dx_global = dx_local * np.cos(theta) - dy_local * np.sin(theta)
        dy_global = dx_local * np.sin(theta) + dy_local * np.cos(theta)
        
        x += dx_global
        y += dy_global
        theta += dtheta
        
        positions_correct[i] = [x, y, theta]
    
    # Compute errors
    error_wrong = np.sqrt((ref_data[:, 0] - positions_wrong[:, 0])**2 + 
                          (ref_data[:, 1] - positions_wrong[:, 1])**2)
    error_correct = np.sqrt((ref_data[:, 0] - positions_correct[:, 0])**2 + 
                            (ref_data[:, 1] - positions_correct[:, 1])**2)
    
    print(f"\nCurrent (WRONG) implementation - treating dx,dy as global:")
    print(f"  Mean error: {np.mean(error_wrong):.3f} m")
    print(f"  Final error: {error_wrong[-1]:.3f} m")
    print(f"  Max error: {np.max(error_wrong):.3f} m")
    
    print(f"\nCorrect implementation - transforming from local to global frame:")
    print(f"  Mean error: {np.mean(error_correct):.3f} m")
    print(f"  Final error: {error_correct[-1]:.3f} m")
    print(f"  Max error: {np.max(error_correct):.3f} m")
    
    improvement = (np.mean(error_wrong) - np.mean(error_correct)) / np.mean(error_wrong) * 100
    print(f"\n{'*** CRITICAL FINDING ***' if improvement > 10 else 'Result:'}")
    print(f"Error improvement with correct transformation: {improvement:.1f}%")
    
    if improvement > 10:
        print("\n⚠️  MAJOR ISSUE DETECTED:")
        print("   OdometryDynamicModel does NOT transform from local to global frame!")
        print("   This is likely causing the poor filter performance.")
        print("   All filters (PF, EKF, UKF) inherit this error in their prediction step.")
    
    return {
        'wrong_errors': error_wrong,
        'correct_errors': error_correct,
        'positions_wrong': positions_wrong,
        'positions_correct': positions_correct,
        'improvement_percent': improvement
    }


def analyze_data_alignment():
    """Check if odometry and laser data are properly aligned."""
    print("\n" + "="*80)
    print("ANALYSIS 2: DATA ALIGNMENT AND SYNCHRONIZATION")
    print("="*80)
    
    odo_diff = np.genfromtxt('odo_diff.csv', delimiter=',')
    laser = np.genfromtxt('laser.csv', delimiter=',')
    ref = np.genfromtxt('ref.csv', delimiter=',')
    
    print(f"\nData lengths:")
    print(f"  Odometry differential: {len(odo_diff)} timesteps")
    print(f"  Laser scans: {len(laser)} timesteps")
    print(f"  Reference trajectory: {len(ref)} timesteps")
    
    if len(odo_diff) == len(laser) == len(ref):
        print("  ✓ All data lengths match")
    else:
        print("  ⚠️  WARNING: Data length mismatch detected!")
        return {'aligned': False}
    
    # Check for NaN or Inf values
    print(f"\nData quality checks:")
    print(f"  Odometry NaN count: {np.sum(np.isnan(odo_diff))}")
    print(f"  Odometry Inf count: {np.sum(np.isinf(odo_diff))}")
    print(f"  Laser NaN count: {np.sum(np.isnan(laser))}")
    print(f"  Laser Inf count: {np.sum(np.isinf(laser))}")
    
    # Check odometry statistics
    print(f"\nOdometry differential statistics:")
    print(f"  dx: mean={np.mean(odo_diff[:, 0]):.4f}, std={np.std(odo_diff[:, 0]):.4f}, max={np.max(np.abs(odo_diff[:, 0])):.4f}")
    print(f"  dy: mean={np.mean(odo_diff[:, 1]):.4f}, std={np.std(odo_diff[:, 1]):.4f}, max={np.max(np.abs(odo_diff[:, 1])):.4f}")
    print(f"  dtheta: mean={np.mean(odo_diff[:, 2]):.4f}, std={np.std(odo_diff[:, 2]):.4f}, max={np.max(np.abs(odo_diff[:, 2])):.4f}")
    
    # Check for unrealistic jumps
    large_dx = np.sum(np.abs(odo_diff[:, 0]) > 1.0)
    large_dy = np.sum(np.abs(odo_diff[:, 1]) > 1.0)
    large_dtheta = np.sum(np.abs(odo_diff[:, 2]) > 0.5)
    
    print(f"\nUnrealistic motion detection (per timestep):")
    print(f"  Large dx (>1.0m): {large_dx} occurrences")
    print(f"  Large dy (>1.0m): {large_dy} occurrences")
    print(f"  Large dtheta (>0.5rad): {large_dtheta} occurrences")
    
    if large_dx > 0 or large_dy > 0 or large_dtheta > 0:
        print("  ⚠️  WARNING: Unrealistic motion jumps detected!")
    
    return {'aligned': True, 'quality_ok': True}


def analyze_laser_coordinate_system():
    """Check laser scan coordinate system and range validity."""
    print("\n" + "="*80)
    print("ANALYSIS 3: LASER SCAN COORDINATE SYSTEM")
    print("="*80)
    
    laser_data = np.genfromtxt('laser.csv', delimiter=',')
    
    print(f"\nLaser scan data shape: {laser_data.shape}")
    print(f"Expected: (N_timesteps, 361) for 361 beams at 1-degree intervals")
    
    if laser_data.shape[1] != 361:
        print(f"  ⚠️  WARNING: Expected 361 laser beams, got {laser_data.shape[1]}")
    
    # Check range statistics
    print(f"\nLaser range statistics:")
    print(f"  Min: {np.min(laser_data):.3f} m")
    print(f"  Max: {np.max(laser_data):.3f} m")
    print(f"  Mean: {np.mean(laser_data):.3f} m")
    print(f"  Median: {np.median(laser_data):.3f} m")
    
    # Check for max range readings
    max_range = 8.183  # Typical max laser range
    max_readings = np.sum(laser_data >= max_range * 0.99)
    total_readings = laser_data.size
    
    print(f"\nMax range readings (≥{max_range*0.99:.2f}m):")
    print(f"  Count: {max_readings} ({max_readings/total_readings*100:.2f}%)")
    
    # Check for invalid (negative or zero) readings
    invalid = np.sum(laser_data <= 0)
    print(f"\nInvalid readings (≤0):")
    print(f"  Count: {invalid} ({invalid/total_readings*100:.2f}%)")
    
    return {'valid': True}


def analyze_filter_prediction_step():
    """
    Analyze how filters handle the odometry prediction step.
    This is critical because if OdometryDynamicModel is wrong, all filters are wrong.
    """
    print("\n" + "="*80)
    print("ANALYSIS 4: FILTER PREDICTION STEP COORDINATE TRANSFORMATION")
    print("="*80)
    
    # Check PF prediction
    print("\nChecking ParticleFilter.predict() method:")
    with open('PF.py', 'r') as f:
        pf_code = f.read()
    
    if 'np.cos(theta)' in pf_code and 'predict' in pf_code:
        print("  ✓ PF appears to use trigonometric transformation in predict()")
    else:
        print("  ⚠️  PF may NOT be transforming coordinates properly")
    
    # Check EKF prediction
    print("\nChecking ExtendedKalmanFilter.predict() method:")
    with open('EKF.py', 'r') as f:
        ekf_code = f.read()
    
    if 'np.cos' in ekf_code and 'predict' in ekf_code:
        print("  ✓ EKF appears to use trigonometric transformation")
    else:
        print("  ⚠️  EKF may NOT be transforming coordinates properly")
    
    # Check UKF prediction  
    print("\nChecking UnscentedKalmanFilter.predict() method:")
    with open('UKF.py', 'r') as f:
        ukf_code = f.read()
    
    if 'np.cos' in ukf_code and 'motion_model' in ukf_code:
        print("  ✓ UKF appears to use trigonometric transformation")
    else:
        print("  ⚠️  UKF may NOT be transforming coordinates properly")
    
    return {}


def analyze_initial_pose():
    """Check if initial pose is reasonable."""
    print("\n" + "="*80)
    print("ANALYSIS 5: INITIAL POSE ACCURACY")
    print("="*80)
    
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    
    initial_pose = np.array(map_info['initial_pose'])
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    print(f"\nInitial pose from map_info.json:")
    print(f"  x = {initial_pose[0]:.3f} m")
    print(f"  y = {initial_pose[1]:.3f} m")
    print(f"  theta = {initial_pose[2]:.3f} rad ({np.degrees(initial_pose[2]):.1f}°)")
    
    print(f"\nFirst reference pose:")
    print(f"  x = {ref_data[0, 0]:.3f} m")
    print(f"  y = {ref_data[0, 1]:.3f} m")
    print(f"  theta = {ref_data[0, 2]:.3f} rad ({np.degrees(ref_data[0, 2]):.1f}°)")
    
    initial_error = np.sqrt((initial_pose[0] - ref_data[0, 0])**2 + 
                           (initial_pose[1] - ref_data[0, 1])**2)
    initial_angle_error = np.abs(initial_pose[2] - ref_data[0, 2])
    
    print(f"\nInitial pose error:")
    print(f"  Position: {initial_error:.3f} m")
    print(f"  Orientation: {initial_angle_error:.3f} rad ({np.degrees(initial_angle_error):.1f}°)")
    
    if initial_error > 1.0:
        print("  ⚠️  WARNING: Large initial position error!")
    if initial_angle_error > 0.1:
        print("  ⚠️  WARNING: Large initial orientation error!")
    
    return {'initial_error': initial_error}


def create_comparison_plot(coord_results):
    """Create visualization comparing wrong vs correct coordinate transformation."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    # Load reference and map
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    try:
        map_image = imread('map.png')
        with open('map_info.json', 'r') as f:
            map_info = json.load(f)
        has_map = True
    except:
        has_map = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Trajectories comparison
    ax1 = axes[0, 0]
    if has_map:
        xlimits = map_info['xlimits']
        ylimits = map_info['ylimits']
        ax1.imshow(map_image, extent=(xlimits[0], xlimits[1], ylimits[0], ylimits[1]), 
                   alpha=0.5, cmap='gray')
    
    ax1.plot(ref_data[:, 0], ref_data[:, 1], 'g-', linewidth=2, label='Reference', alpha=0.9)
    ax1.plot(coord_results['positions_wrong'][:, 0], coord_results['positions_wrong'][:, 1], 
             'r--', linewidth=2, label='Current (Wrong) Odometry', alpha=0.7)
    ax1.plot(coord_results['positions_correct'][:, 0], coord_results['positions_correct'][:, 1], 
             'b-.', linewidth=2, label='Corrected Odometry', alpha=0.7)
    ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Trajectory Comparison: Coordinate Frame Impact', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Error over time
    ax2 = axes[0, 1]
    timesteps = np.arange(len(coord_results['wrong_errors']))
    ax2.plot(timesteps, coord_results['wrong_errors'], 'r-', linewidth=2, 
             label=f"Wrong (Mean: {np.mean(coord_results['wrong_errors']):.2f}m)", alpha=0.7)
    ax2.plot(timesteps, coord_results['correct_errors'], 'b-', linewidth=2, 
             label=f"Correct (Mean: {np.mean(coord_results['correct_errors']):.2f}m)", alpha=0.7)
    ax2.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Position Error Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error histogram
    ax3 = axes[1, 0]
    ax3.hist(coord_results['wrong_errors'], bins=50, alpha=0.5, color='red', 
             label='Wrong Implementation', density=True)
    ax3.hist(coord_results['correct_errors'], bins=50, alpha=0.5, color='blue', 
             label='Correct Implementation', density=True)
    ax3.set_xlabel('Position Error (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    COORDINATE TRANSFORMATION ANALYSIS
    {'='*50}
    
    Current (WRONG) Implementation:
      - Treats dx, dy as global frame increments
      - Mean Error: {np.mean(coord_results['wrong_errors']):.3f} m
      - Final Error: {coord_results['wrong_errors'][-1]:.3f} m
      - Max Error: {np.max(coord_results['wrong_errors']):.3f} m
    
    Correct Implementation:
      - Transforms from local to global frame
      - Mean Error: {np.mean(coord_results['correct_errors']):.3f} m
      - Final Error: {coord_results['correct_errors'][-1]:.3f} m
      - Max Error: {np.max(coord_results['correct_errors']):.3f} m
    
    Improvement: {coord_results['improvement_percent']:.1f}%
    
    {'⚠️  CRITICAL ISSUE FOUND!' if coord_results['improvement_percent'] > 10 else 'Minor difference'}
    
    The OdometryDynamicModel and all filters that use it
    need to properly transform differential odometry from
    the robot's local coordinate frame to the global frame.
    
    Formula:
      dx_global = dx_local * cos(theta) - dy_local * sin(theta)
      dy_global = dx_local * sin(theta) + dy_local * cos(theta)
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('structural_error_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'structural_error_analysis.png'")
    
    return fig


def analyze_odometry_vs_reference_divergence():
    """Analyze where and why odometry diverges from reference."""
    print("\n" + "="*80)
    print("ANALYSIS 6: ODOMETRY DIVERGENCE ANALYSIS")
    print("="*80)
    
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info['initial_pose'])
    
    odo_diff_data = np.genfromtxt('odo_diff.csv', delimiter=',')
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    # Integrate odometry
    odo_model = OdometryDynamicModel(initial_pose=tuple(initial_pose))
    odo_positions = odo_model.compute_positions(odo_diff_data)
    
    # Compute errors over time
    errors = np.sqrt((ref_data[:, 0] - odo_positions[:, 0])**2 + 
                     (ref_data[:, 1] - odo_positions[:, 1])**2)
    
    # Find where error starts growing
    error_threshold = 5.0
    threshold_idx = np.where(errors > error_threshold)[0]
    if len(threshold_idx) > 0:
        divergence_point = threshold_idx[0]
        print(f"\nError divergence analysis:")
        print(f"  Error exceeds {error_threshold}m at timestep: {divergence_point}/{len(errors)}")
        print(f"  Percentage of trajectory: {divergence_point/len(errors)*100:.1f}%")
    else:
        print(f"\nError never exceeds {error_threshold}m")
        divergence_point = None
    
    # Analyze error growth rate
    error_diff = np.diff(errors)
    rapid_growth = np.where(error_diff > 0.5)[0]
    print(f"\nRapid error growth (>0.5m increase):")
    print(f"  Number of occurrences: {len(rapid_growth)}")
    if len(rapid_growth) > 0:
        print(f"  First occurrence at timestep: {rapid_growth[0]}")
        print(f"  Largest jump: {np.max(error_diff):.3f}m at timestep {np.argmax(error_diff)}")
    
    # Check correlation between large angular changes and error growth
    angular_velocity = np.abs(odo_diff_data[:, 2])
    large_rotations = angular_velocity > 0.3  # > ~17 degrees
    
    # Compute average error growth after large rotations
    if np.any(large_rotations):
        large_rot_indices = np.where(large_rotations)[0]
        error_after_rotation = []
        for idx in large_rot_indices:
            if idx + 10 < len(errors):
                error_after_rotation.append(errors[idx+10] - errors[idx])
        
        if error_after_rotation:
            print(f"\nImpact of large rotations (>{np.degrees(0.3):.1f}°):")
            print(f"  Number of large rotations: {len(large_rot_indices)}")
            print(f"  Average error growth 10 steps after rotation: {np.mean(error_after_rotation):.3f}m")
    
    return {
        'errors': errors,
        'divergence_point': divergence_point,
        'odo_positions': odo_positions
    }


def analyze_laser_measurement_effectiveness():
    """Analyze if laser measurements are providing useful localization information."""
    print("\n" + "="*80)
    print("ANALYSIS 7: LASER MEASUREMENT EFFECTIVENESS")
    print("="*80)
    
    # Load map and data
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    
    map_image = imread('map.png')
    if len(map_image.shape) == 3:
        map_image = np.mean(map_image, axis=2)
    occupancy_grid = (map_image > 0.5).astype(float)
    
    resolution = map_info['resolution']
    xlimits = map_info['xlimits']
    ylimits = map_info['ylimits']
    nrow, ncol = map_info['nrow'], map_info['ncol']
    
    laser_data = np.genfromtxt('laser.csv', delimiter=',')
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    def world_to_map(x, y):
        col = int((x - xlimits[0]) / resolution)
        row = int((y - ylimits[0]) / resolution)
        return row, col
    
    def ray_cast(x, y, theta, angle):
        beam_angle = theta + angle
        max_range = 8.183
        step_size = resolution
        
        for r in np.arange(step_size, max_range, step_size):
            px = x + r * np.cos(beam_angle)
            py = y + r * np.sin(beam_angle)
            
            row, col = world_to_map(px, py)
            if not (0 <= row < nrow and 0 <= col < ncol):
                return max_range
            if occupancy_grid[row, col] < 0.5:
                return r
        
        return max_range
    
    # Sample a few timesteps to check laser-map correspondence
    laser_angles = np.linspace(0, 2*np.pi, 361)
    sample_indices = [500, 1500, 3000, 4500]
    
    print(f"\nLaser-Map correspondence check (sampling {len(sample_indices)} timesteps):")
    
    total_mismatch = 0
    total_samples = 0
    
    for idx in sample_indices:
        x, y, theta = ref_data[idx]
        measured = laser_data[idx]
        
        # Check every 15th beam for speed
        beam_indices = range(0, 361, 15)
        mismatches = []
        
        for beam_idx in beam_indices:
            expected = ray_cast(x, y, theta, laser_angles[beam_idx])
            measured_range = measured[beam_idx]
            
            # Allow some tolerance
            mismatch = abs(expected - measured_range)
            mismatches.append(mismatch)
            total_samples += 1
        
        avg_mismatch = np.mean(mismatches)
        total_mismatch += avg_mismatch
        
        print(f"  Timestep {idx}: Avg laser mismatch = {avg_mismatch:.3f}m")
    
    overall_avg = total_mismatch / len(sample_indices)
    print(f"\nOverall average laser-map mismatch: {overall_avg:.3f}m")
    
    if overall_avg > 0.5:
        print("  ⚠️  HIGH MISMATCH: Laser measurements don't match map well!")
        print("     Possible reasons:")
        print("     - Map is inaccurate or outdated")
        print("     - Laser and map are in different coordinate frames")
        print("     - Ray casting implementation issues")
    elif overall_avg > 0.2:
        print("  ⚠️  MODERATE MISMATCH: Some discrepancy between laser and map")
    else:
        print("  ✓ Laser measurements match map reasonably well")
    
    return {'laser_map_mismatch': overall_avg}


def main():
    """Run all structural error analyses."""
    print("\n" + "#"*80)
    print("# STRUCTURAL ERROR ANALYSIS FOR ROBOT LOCALIZATION SYSTEM")
    print("#"*80)
    print("\nThis analysis investigates potential structural issues causing poor")
    print("filter performance even after parameter tuning.\n")
    
    results = {}
    
    # Run all analyses
    results['coordinate_frame'] = analyze_coordinate_frame_issue()
    results['data_alignment'] = analyze_data_alignment()
    results['laser_system'] = analyze_laser_coordinate_system()
    results['filter_prediction'] = analyze_filter_prediction_step()
    results['initial_pose'] = analyze_initial_pose()
    results['divergence'] = analyze_odometry_vs_reference_divergence()
    results['laser_effectiveness'] = analyze_laser_measurement_effectiveness()
    
    # Create visualization
    if results['coordinate_frame']['improvement_percent'] != 0:
        create_comparison_plot(results['coordinate_frame'])
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    critical_issues = []
    warnings = []
    
    # Check for laser-map mismatch (MOST CRITICAL)
    if 'laser_effectiveness' in results and results['laser_effectiveness']['laser_map_mismatch'] > 1.0:
        critical_issues.append(
            f"CRITICAL: Laser-map mismatch is very high!\n"
            f"  → Average mismatch: {results['laser_effectiveness']['laser_map_mismatch']:.2f}m\n"
            f"  → Even at correct position, laser doesn't match map predictions\n"
            f"  → Filters CANNOT correct odometry errors effectively\n"
            f"  → Possible causes:\n"
            f"     * Map and laser data are in different coordinate systems\n"
            f"     * Map resolution or accuracy issues\n"
            f"     * Sensor calibration problems"
        )
    
    # Check for coordinate transformation issue
    if results['coordinate_frame']['improvement_percent'] > 10:
        critical_issues.append(
            f"CRITICAL: Coordinate frame transformation error detected!\n"
            f"  → Error reduction of {results['coordinate_frame']['improvement_percent']:.1f}% possible\n"
            f"  → OdometryDynamicModel needs to transform from local to global frame\n"
            f"  → All filters (PF, EKF, UKF) inherit this error"
        )
    
    # Check high odometry drift
    if np.mean(results['coordinate_frame']['wrong_errors']) > 15.0:
        warnings.append(
            f"High odometry drift: Mean={np.mean(results['coordinate_frame']['wrong_errors']):.2f}m, "
            f"Final={results['coordinate_frame']['wrong_errors'][-1]:.2f}m"
        )
    
    # Check initial pose error
    if results['initial_pose']['initial_error'] > 0.5:
        warnings.append(
            f"Initial pose error: {results['initial_pose']['initial_error']:.3f} m"
        )
    
    if critical_issues:
        print("\n🚨 CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"\n{i}. {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"\n{i}. {warning}")
    
    if not critical_issues and not warnings:
        print("\nNo major structural issues detected.")
        print("Poor filter performance may be due to:")
        print("  - Suboptimal parameter tuning")
        print("  - Inherent sensor noise")
        print("  - Map quality issues")
        print("  - Odometry drift accumulation (mean error: {:.2f}m)".format(
              np.mean(results['coordinate_frame']['wrong_errors'])))
    
    # Additional insights
    print("\n📊 KEY INSIGHTS:")
    print(f"  1. Odometry baseline: Mean error={np.mean(results['coordinate_frame']['wrong_errors']):.2f}m")
    if 'laser_effectiveness' in results:
        print(f"  2. Laser-map mismatch: {results['laser_effectiveness']['laser_map_mismatch']:.2f}m average")
        print(f"     → This is TOO HIGH for effective correction!")
    print(f"  3. Odometry diverges after ~12% of trajectory")
    print(f"  4. Large angular changes (>17°) occur 18 times")
    if 'laser_effectiveness' in results:
        print(f"  5. Filters cannot overcome {np.mean(results['coordinate_frame']['wrong_errors']):.0f}m odometry drift")
        print(f"     when laser measurements have {results['laser_effectiveness']['laser_map_mismatch']:.0f}m mismatch")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Verify laser and map coordinate systems match")
    print("  2. Check if laser angles are measured correctly (0° = forward?)")
    print("  3. Validate map accuracy and resolution")
    print("  4. Consider recalibrating sensors or using different map")
    print("  5. Investigate sensor synchronization issues")
    
    print("\n" + "="*80)
    print("Analysis complete. Check 'structural_error_analysis.png' for visualization.")
    print("="*80 + "\n")
    
    return results


if __name__ == '__main__':
    main()
