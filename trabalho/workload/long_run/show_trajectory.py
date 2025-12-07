import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from OdometryStaticModel import OdometryStaticModel
from OdometryDynamicModel import OdometryDynamicModel
import os
import json

# Read the reference trajectory CSV file
ref_data = np.genfromtxt('ref.csv', delimiter=',')
ref_x = ref_data[:, 0]
ref_y = ref_data[:, 1]
ref_angle = ref_data[:, 2]

"""
Load and compute trajectories:
- Static (cumulative) odometry from 'odo.csv' using OdometryStaticModel
- Dynamic (differential) odometry from 'odo_diff.csv' using OdometryDynamicModel
"""
initial_pose = (ref_x[0], ref_y[0], ref_angle[0])

# Static trajectory from cumulative odometry
odo_static_model = OdometryStaticModel(initial_pose=initial_pose)
odo_cum_data = OdometryStaticModel.load_from_csv('odo.csv')
odo_static_positions = odo_static_model.compute_positions(odo_cum_data)
odo_x = odo_static_positions[:, 0]
odo_y = odo_static_positions[:, 1]
odo_angle = odo_static_positions[:, 2]

# Dynamic trajectory from differential odometry
odo_dyn_model = OdometryDynamicModel(initial_pose=initial_pose)
odo_diff_data = OdometryDynamicModel.load_from_csv('odo_diff.csv')
odo_dyn_positions = odo_dyn_model.compute_positions(odo_diff_data)
odo_dyn_x = odo_dyn_positions[:, 0]
odo_dyn_y = odo_dyn_positions[:, 1]
odo_dyn_angle = odo_dyn_positions[:, 2]

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 12))

# Load and display the map if available
map_file = 'map.png'
if os.path.exists(map_file):
    try:
        map_image = imread(map_file)
        
        # Display map as background
        # Map coordinates need to be inferred from the image dimensions
        # Assuming the map image corresponds to the robot's environment space
        map_extent = None
        
        # Read map extent from image metadata, that needs to be available
        with open('map_info.json', 'r') as f:
            map_info = json.load(f)

        xlimits = map_info['xlimits']
        ylimits = map_info['ylimits']
        map_extent = (xlimits[0], xlimits[1], ylimits[0], ylimits[1])
        
        # Display map with appropriate extent
        ax.imshow(map_image, extent=map_extent, alpha=0.6, cmap='gray')
    except Exception as e:
        print(f"Warning: Could not load map image: {e}")
        print("Continuing without map background...")
else:
    print(f"Map file '{map_file}' not found. Displaying trajectories without map.")

# Plot reference trajectory
ax.plot(ref_x, ref_y, color='lime', label='Reference Trajectory', linewidth=2.5, alpha=0.9)

# Plot odometry trajectories
ax.plot(odo_x, odo_y, color='red', label='Static Odometry', linewidth=2.5, alpha=0.9)
ax.plot(odo_dyn_x, odo_dyn_y, color='orange', label='Dynamic Odometry', linewidth=2.0, alpha=0.9, linestyle='--')

# Plot orientation arrows at regular intervals for reference
arrow_interval = max(1, len(ref_x) // 40)
for i in range(0, len(ref_x), arrow_interval):
    dx = np.cos(ref_angle[i]) * 0.5
    dy = np.sin(ref_angle[i]) * 0.5
    ax.arrow(ref_x[i], ref_y[i], dx, dy, head_width=0.4, head_length=0.6,
             fc='lime', ec='lime', alpha=0.8, linewidth=1.5)

# Plot orientation arrows for static odometry
for i in range(0, len(odo_x), arrow_interval):
    dx = np.cos(odo_angle[i]) * 0.5
    dy = np.sin(odo_angle[i]) * 0.5
    ax.arrow(odo_x[i], odo_y[i], dx, dy, head_width=0.4, head_length=0.6,
             fc='red', ec='red', alpha=0.8, linewidth=1.5)

# Plot orientation arrows for dynamic odometry
for i in range(0, len(odo_dyn_x), arrow_interval):
    dx = np.cos(odo_dyn_angle[i]) * 0.5
    dy = np.sin(odo_dyn_angle[i]) * 0.5
    ax.arrow(odo_dyn_x[i], odo_dyn_y[i], dx, dy, head_width=0.4, head_length=0.6,
             fc='orange', ec='orange', alpha=0.7, linewidth=1.2)

# Mark start and end points
ax.plot(ref_x[0], ref_y[0], 'go', markersize=14, label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
ax.plot(ref_x[-1], ref_y[-1], 'ms', markersize=14, label='Ref End', zorder=5, markeredgecolor='purple', markeredgewidth=2)
ax.plot(odo_x[-1], odo_y[-1], 'c^', markersize=14, label='Static Odo End', zorder=5, markeredgecolor='darkblue', markeredgewidth=2)
ax.plot(odo_dyn_x[-1], odo_dyn_y[-1], 'yv', markersize=14, label='Dynamic Odo End', zorder=5, markeredgecolor='brown', markeredgewidth=2)

# Labels and formatting
ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
ax.set_title('Robot Trajectory Comparison: Reference vs Odometry', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.95)
ax.axis('equal')

"""Error statistics for both odometry trajectories"""
final_error_static = np.sqrt((ref_x[-1] - odo_x[-1])**2 + (ref_y[-1] - odo_y[-1])**2)
final_error_dynamic = np.sqrt((ref_x[-1] - odo_dyn_x[-1])**2 + (ref_y[-1] - odo_dyn_y[-1])**2)

position_errors_static = np.sqrt((ref_x - odo_x)**2 + (ref_y - odo_y)**2)
position_errors_dynamic = np.sqrt((ref_x - odo_dyn_x)**2 + (ref_y - odo_dyn_y)**2)

mean_error_static = np.mean(position_errors_static)
max_error_static = np.max(position_errors_static)
mean_error_dynamic = np.mean(position_errors_dynamic)
max_error_dynamic = np.max(position_errors_dynamic)

stats_text = (
    f'Static Odo — Final: {final_error_static:.3f} m, Mean: {mean_error_static:.3f} m, Max: {max_error_static:.3f} m\n'
    f'Dynamic Odo — Final: {final_error_dynamic:.3f} m, Mean: {mean_error_dynamic:.3f} m, Max: {max_error_dynamic:.3f} m'
)

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='orange', linewidth=2))

plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Trajectories plotted and saved to 'trajectory_comparison.png'")
print(f"Reference trajectory: {len(ref_x)} points")
print(f"Static odometry trajectory: {len(odo_x)} points")
print(f"Dynamic odometry trajectory: {len(odo_dyn_x)} points")
print(f"Static — Final: {final_error_static:.3f} m, Mean: {mean_error_static:.3f} m, Max: {max_error_static:.3f} m")
print(f"Dynamic — Final: {final_error_dynamic:.3f} m, Mean: {mean_error_dynamic:.3f} m, Max: {max_error_dynamic:.3f} m")
