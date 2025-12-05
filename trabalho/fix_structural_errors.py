"""
Fix Structural Errors in Robot Localization Data

This script attempts to fix structural issues identified in the data:
1. Laser-map coordinate system misalignment
2. Potential laser angle offset
3. Data synchronization issues

The script will test various fixes and save corrected data files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
import os
from scipy.optimize import minimize_scalar
from scipy.ndimage import distance_transform_edt


def load_data():
    """Load all data files."""
    print("Loading data files...")
    
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    
    map_image = imread('map.png')
    if len(map_image.shape) == 3:
        map_image = np.mean(map_image, axis=2)
    occupancy_grid = (map_image > 0.5).astype(float)
    
    odo_diff = np.genfromtxt('odo_diff.csv', delimiter=',')
    laser = np.genfromtxt('laser.csv', delimiter=',')
    ref = np.genfromtxt('ref.csv', delimiter=',')
    
    return {
        'map_info': map_info,
        'occupancy_grid': occupancy_grid,
        'odo_diff': odo_diff,
        'laser': laser,
        'ref': ref
    }


def world_to_map(x, y, map_info):
    """Convert world coordinates to map grid indices."""
    resolution = map_info['resolution']
    xlimits = map_info['xlimits']
    ylimits = map_info['ylimits']
    
    col = int((x - xlimits[0]) / resolution)
    row = int((y - ylimits[0]) / resolution)
    return row, col


def ray_cast(x, y, theta, angle, occupancy_grid, map_info, max_range=8.183):
    """Ray casting to get expected laser range."""
    beam_angle = theta + angle
    resolution = map_info['resolution']
    nrow, ncol = map_info['nrow'], map_info['ncol']
    
    step_size = resolution
    
    for r in np.arange(step_size, max_range, step_size):
        px = x + r * np.cos(beam_angle)
        py = y + r * np.sin(beam_angle)
        
        row, col = world_to_map(px, py, map_info)
        if not (0 <= row < nrow and 0 <= col < ncol):
            return max_range
        if occupancy_grid[row, col] < 0.5:
            return r
    
    return max_range


def compute_laser_map_mismatch(laser_data, ref_data, occupancy_grid, map_info, angle_offset=0.0):
    """Compute average laser-map mismatch with given angle offset."""
    laser_angles = np.linspace(0, 2*np.pi, 361) + angle_offset
    
    # Sample timesteps to speed up computation
    sample_indices = np.linspace(100, len(ref_data)-100, 20, dtype=int)
    
    total_error = 0.0
    count = 0
    
    for idx in sample_indices:
        x, y, theta = ref_data[idx]
        measured = laser_data[idx]
        
        # Check every 15th beam
        for beam_idx in range(0, 361, 15):
            expected = ray_cast(x, y, theta, laser_angles[beam_idx], occupancy_grid, map_info)
            measured_range = measured[beam_idx]
            
            # Ignore max range readings
            if measured_range < 8.0 and expected < 8.0:
                error = abs(expected - measured_range)
                total_error += error
                count += 1
    
    return total_error / count if count > 0 else float('inf')


def find_optimal_laser_angle_offset(data):
    """Find the optimal laser angle offset to minimize laser-map mismatch."""
    print("\n" + "="*80)
    print("SEARCHING FOR OPTIMAL LASER ANGLE OFFSET")
    print("="*80)
    
    print("\nTesting different angle offsets to minimize laser-map mismatch...")
    
    def objective(angle_offset):
        mismatch = compute_laser_map_mismatch(
            data['laser'], data['ref'], data['occupancy_grid'], 
            data['map_info'], angle_offset
        )
        return mismatch
    
    # Test a range of offsets
    test_offsets = np.linspace(-np.pi, np.pi, 25)
    test_errors = []
    
    print("Testing angle offsets from -180° to +180°...")
    for offset in test_offsets:
        error = objective(offset)
        test_errors.append(error)
        if abs(np.degrees(offset)) % 45 < 1:  # Print every 45 degrees
            print(f"  Offset {np.degrees(offset):6.1f}°: Mismatch = {error:.3f}m")
    
    # Find minimum
    best_idx = np.argmin(test_errors)
    best_offset = test_offsets[best_idx]
    best_error = test_errors[best_idx]
    
    # Refine around minimum
    print(f"\nRefining search around {np.degrees(best_offset):.1f}°...")
    result = minimize_scalar(
        objective, 
        bounds=(best_offset - 0.5, best_offset + 0.5),
        method='bounded'
    )
    
    optimal_offset = result.x
    optimal_error = result.fun
    
    print(f"\n✓ Optimal angle offset found:")
    print(f"  Offset: {np.degrees(optimal_offset):.2f}° ({optimal_offset:.4f} rad)")
    print(f"  Mismatch: {optimal_error:.3f}m")
    
    # Compare with original
    original_error = compute_laser_map_mismatch(
        data['laser'], data['ref'], data['occupancy_grid'], 
        data['map_info'], 0.0
    )
    
    improvement = (original_error - optimal_error) / original_error * 100
    print(f"\nOriginal mismatch: {original_error:.3f}m")
    print(f"Improved mismatch: {optimal_error:.3f}m")
    print(f"Improvement: {improvement:.1f}%")
    
    return optimal_offset, optimal_error, improvement


def apply_laser_angle_correction(laser_data, angle_offset):
    """Apply angle correction by rotating laser beam indices."""
    print(f"\nApplying laser angle correction of {np.degrees(angle_offset):.2f}°...")
    
    n_beams = 361
    corrected_laser = np.zeros_like(laser_data)
    
    # Calculate shift in beam indices
    shift_beams = int(round(angle_offset / (2*np.pi) * n_beams))
    
    print(f"  Rotating laser readings by {shift_beams} beam indices")
    
    # Circular shift
    for t in range(len(laser_data)):
        corrected_laser[t] = np.roll(laser_data[t], shift_beams)
    
    return corrected_laser


def test_odometry_coordinate_transformation(data):
    """Test if odometry should be transformed from local to global frame."""
    print("\n" + "="*80)
    print("TESTING ODOMETRY COORDINATE TRANSFORMATION")
    print("="*80)
    
    initial_pose = np.array(data['map_info']['initial_pose'])
    odo_diff = data['odo_diff']
    ref = data['ref']
    
    # Current: assume global frame
    positions_global = np.zeros_like(odo_diff)
    positions_global[0] = initial_pose
    x, y, theta = initial_pose
    
    for i in range(1, len(odo_diff)):
        x += odo_diff[i, 0]
        y += odo_diff[i, 1]
        theta += odo_diff[i, 2]
        positions_global[i] = [x, y, theta]
    
    error_global = np.sqrt((ref[:, 0] - positions_global[:, 0])**2 + 
                           (ref[:, 1] - positions_global[:, 1])**2)
    
    # Alternative: transform from local to global frame
    positions_local = np.zeros_like(odo_diff)
    positions_local[0] = initial_pose
    x, y, theta = initial_pose
    
    for i in range(1, len(odo_diff)):
        dx_local = odo_diff[i, 0]
        dy_local = odo_diff[i, 1]
        dtheta = odo_diff[i, 2]
        
        # Transform to global
        dx_global = dx_local * np.cos(theta) - dy_local * np.sin(theta)
        dy_global = dx_local * np.sin(theta) + dy_local * np.cos(theta)
        
        x += dx_global
        y += dy_global
        theta += dtheta
        
        positions_local[i] = [x, y, theta]
    
    error_local = np.sqrt((ref[:, 0] - positions_local[:, 0])**2 + 
                          (ref[:, 1] - positions_local[:, 1])**2)
    
    print(f"\nAssuming odometry is in GLOBAL frame:")
    print(f"  Mean error: {np.mean(error_global):.3f}m")
    print(f"  Final error: {error_global[-1]:.3f}m")
    
    print(f"\nTransforming from LOCAL to GLOBAL frame:")
    print(f"  Mean error: {np.mean(error_local):.3f}m")
    print(f"  Final error: {error_local[-1]:.3f}m")
    
    if np.mean(error_global) < np.mean(error_local):
        print("\n✓ Odometry data is already in GLOBAL frame (current assumption is correct)")
        return None  # No transformation needed
    else:
        print("\n✓ Odometry should be transformed from LOCAL to GLOBAL frame")
        return positions_local


def remove_outlier_laser_readings(laser_data, threshold=8.0):
    """Remove or interpolate outlier laser readings."""
    print("\n" + "="*80)
    print("CLEANING OUTLIER LASER READINGS")
    print("="*80)
    
    cleaned_laser = laser_data.copy()
    n_outliers = 0
    
    # Mark readings at max range as potential outliers
    max_range_mask = laser_data >= threshold
    n_max_range = np.sum(max_range_mask)
    
    print(f"\nMax range readings (>={threshold}m): {n_max_range} ({n_max_range/laser_data.size*100:.2f}%)")
    print("These will be kept as-is (they represent actual max range)")
    
    # Check for sudden spikes within a scan
    for t in range(len(laser_data)):
        scan = cleaned_laser[t].copy()
        
        # Median filter approach: identify readings that differ significantly from neighbors
        for i in range(1, len(scan)-1):
            if scan[i] < threshold:  # Don't "fix" max range readings
                neighbors = [scan[i-1], scan[i+1]]
                valid_neighbors = [n for n in neighbors if n < threshold]
                
                if len(valid_neighbors) >= 1:
                    median_neighbor = np.median(valid_neighbors)
                    if abs(scan[i] - median_neighbor) > 2.0:  # 2m spike threshold
                        # Replace with interpolated value
                        cleaned_laser[t, i] = median_neighbor
                        n_outliers += 1
    
    print(f"Outlier spikes detected and interpolated: {n_outliers}")
    
    return cleaned_laser


def create_comparison_visualization(data, corrected_laser, angle_offset):
    """Create visualization comparing original vs corrected data."""
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*80)
    
    ref = data['ref']
    map_info = data['map_info']
    occupancy_grid = data['occupancy_grid']
    
    # Sample timestep for detailed comparison
    sample_idx = 1000
    
    laser_angles_original = np.linspace(0, 2*np.pi, 361)
    laser_angles_corrected = laser_angles_original + angle_offset
    
    x, y, theta = ref[sample_idx]
    original_scan = data['laser'][sample_idx]
    corrected_scan = corrected_laser[sample_idx]
    
    # Compute expected ranges
    expected_original = np.array([
        ray_cast(x, y, theta, angle, occupancy_grid, map_info)
        for angle in laser_angles_original
    ])
    expected_corrected = np.array([
        ray_cast(x, y, theta, angle, occupancy_grid, map_info)
        for angle in laser_angles_corrected
    ])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Original laser vs expected
    ax1 = axes[0, 0]
    beam_angles_deg = np.degrees(laser_angles_original)
    ax1.plot(beam_angles_deg, original_scan, 'b-', alpha=0.6, label='Measured', linewidth=1)
    ax1.plot(beam_angles_deg, expected_original, 'r--', alpha=0.6, label='Expected (Map)', linewidth=1)
    ax1.set_xlabel('Beam Angle (degrees)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Range (m)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Original Laser Data (Timestep {sample_idx})', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 8.5])
    
    # Plot 2: Corrected laser vs expected
    ax2 = axes[0, 1]
    ax2.plot(beam_angles_deg, corrected_scan, 'g-', alpha=0.6, label='Corrected', linewidth=1)
    ax2.plot(beam_angles_deg, expected_corrected, 'r--', alpha=0.6, label='Expected (Map)', linewidth=1)
    ax2.set_xlabel('Beam Angle (degrees)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Range (m)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Corrected Laser Data (Offset: {np.degrees(angle_offset):.1f}°)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 8.5])
    
    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    error_original = np.abs(original_scan - expected_original)
    error_corrected = np.abs(corrected_scan - expected_corrected)
    
    # Only consider non-max-range readings
    valid_mask = (original_scan < 8.0) & (expected_original < 8.0)
    
    ax3.hist(error_original[valid_mask], bins=50, alpha=0.6, color='blue', 
             label=f'Original (mean={np.mean(error_original[valid_mask]):.2f}m)', density=True)
    ax3.hist(error_corrected[valid_mask], bins=50, alpha=0.6, color='green', 
             label=f'Corrected (mean={np.mean(error_corrected[valid_mask]):.2f}m)', density=True)
    ax3.set_xlabel('Absolute Error (m)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution (Single Timestep)', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute overall statistics
    original_mismatch = compute_laser_map_mismatch(
        data['laser'], ref, occupancy_grid, map_info, 0.0
    )
    corrected_mismatch = compute_laser_map_mismatch(
        corrected_laser, ref, occupancy_grid, map_info, 0.0
    )
    
    improvement = (original_mismatch - corrected_mismatch) / original_mismatch * 100
    
    summary_text = f"""
    FIX SUMMARY
    {'='*60}
    
    Laser Angle Offset Applied:
      Rotation: {np.degrees(angle_offset):.2f}° ({angle_offset:.4f} rad)
    
    Laser-Map Mismatch:
      Original:  {original_mismatch:.3f} m
      Corrected: {corrected_mismatch:.3f} m
      Improvement: {improvement:.1f}%
    
    Sample Timestep Analysis (t={sample_idx}):
      Original error:  {np.mean(error_original[valid_mask]):.3f} m
      Corrected error: {np.mean(error_corrected[valid_mask]):.3f} m
    
    {'✓ SIGNIFICANT IMPROVEMENT!' if improvement > 20 else '✓ Moderate improvement' if improvement > 5 else '⚠️ Minor improvement'}
    
    The corrected laser data should now better align
    with the map, allowing filters to correct odometry
    drift more effectively.
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fix_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison visualization to 'fix_comparison.png'")
    
    return fig


def save_corrected_data(corrected_laser, suffix='_corrected'):
    """Save corrected laser data to new file."""
    print("\n" + "="*80)
    print("SAVING CORRECTED DATA")
    print("="*80)
    
    output_file = f'laser{suffix}.csv'
    np.savetxt(output_file, corrected_laser, delimiter=',', fmt='%.6f')
    print(f"\nSaved corrected laser data to: {output_file}")
    print(f"  Shape: {corrected_laser.shape}")
    print(f"  Data range: [{np.min(corrected_laser):.3f}, {np.max(corrected_laser):.3f}]")
    
    return output_file


def main():
    """Main execution."""
    print("\n" + "#"*80)
    print("# STRUCTURAL ERROR FIX SCRIPT")
    print("#"*80)
    print("\nThis script attempts to fix structural issues in the robot localization data.")
    print("Fixes attempted:")
    print("  1. Laser angle offset correction")
    print("  2. Odometry coordinate frame check")
    print("  3. Outlier laser reading cleanup")
    print("\n" + "#"*80 + "\n")
    
    # Load data
    data = load_data()
    print(f"✓ Loaded {len(data['ref'])} timesteps of data")
    
    # Test odometry coordinate transformation
    transformed_odo = test_odometry_coordinate_transformation(data)
    if transformed_odo is not None:
        # Save corrected odometry
        print("\nSaving transformed odometry data...")
        # This would require also saving the corrected positions
        print("  Note: Odometry transformation would require regenerating odo_diff from positions")
    
    # Find optimal laser angle offset
    optimal_offset, optimal_error, improvement = find_optimal_laser_angle_offset(data)
    
    # Apply laser corrections
    corrected_laser = apply_laser_angle_correction(data['laser'], optimal_offset)
    
    # Clean outliers
    corrected_laser = remove_outlier_laser_readings(corrected_laser)
    
    # Create comparison visualization
    create_comparison_visualization(data, corrected_laser, optimal_offset)
    
    # Save corrected data
    if improvement > 5:  # Only save if meaningful improvement
        output_file = save_corrected_data(corrected_laser)
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"\n1. The corrected laser data has been saved to '{output_file}'")
        print("2. To use it, update your filter scripts to load this file instead of 'laser.csv'")
        print("3. Re-run the filters with the corrected data")
        print(f"4. Expected improvement in laser-map alignment: {improvement:.1f}%")
        
        if improvement > 20:
            print("\n✓ Significant improvement achieved!")
            print("  The corrected data should substantially improve filter performance.")
        elif improvement > 5:
            print("\n✓ Moderate improvement achieved.")
            print("  The corrected data may improve filter performance.")
        
    else:
        print("\n⚠️  Improvement is minimal (<5%).")
        print("   The structural issues may be due to:")
        print("   - Map inaccuracy")
        print("   - Sensor calibration issues beyond simple angle offset")
        print("   - Environmental changes between map creation and data collection")
        print("   - Not saving corrected data due to low improvement")
    
    print("\n" + "="*80)
    print("Fix script complete. Check 'fix_comparison.png' for detailed results.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
