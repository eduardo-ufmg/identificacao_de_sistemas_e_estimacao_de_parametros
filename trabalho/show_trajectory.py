import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from OdometryModel import OdometryModel
import os
import json

# Read the reference trajectory CSV file
ref_data = np.genfromtxt('ref.csv', delimiter=',')
ref_x = ref_data[:, 0]
ref_y = ref_data[:, 1]
ref_angle = ref_data[:, 2]

# Load and compute odometry positions
odo_model = OdometryModel(initial_pose=(ref_x[0], ref_y[0], ref_angle[0]))
odo_data = OdometryModel.load_from_csv('odo.csv')
odo_positions = odo_model.compute_positions(odo_data)
odo_x = odo_positions[:, 0]
odo_y = odo_positions[:, 1]
odo_angle = odo_positions[:, 2]

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
        ax.text(0.02, 0.02, 'Map overlaid from map.png', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        print(f"Warning: Could not load map image: {e}")
        print("Continuing without map background...")
else:
    print(f"Map file '{map_file}' not found. Displaying trajectories without map.")

# Plot reference trajectory
ax.plot(ref_x, ref_y, color='lime', label='Reference Trajectory', linewidth=2.5, alpha=0.9)

# Plot odometry trajectory
ax.plot(odo_x, odo_y, color='red', label='Odometry Trajectory', linewidth=2.5, alpha=0.9)

# Plot orientation arrows at regular intervals for reference
arrow_interval = max(1, len(ref_x) // 40)
for i in range(0, len(ref_x), arrow_interval):
    dx = np.cos(ref_angle[i]) * 0.5
    dy = np.sin(ref_angle[i]) * 0.5
    ax.arrow(ref_x[i], ref_y[i], dx, dy, head_width=0.4, head_length=0.6,
             fc='lime', ec='lime', alpha=0.8, linewidth=1.5)

# Plot orientation arrows for odometry
for i in range(0, len(odo_x), arrow_interval):
    dx = np.cos(odo_angle[i]) * 0.5
    dy = np.sin(odo_angle[i]) * 0.5
    ax.arrow(odo_x[i], odo_y[i], dx, dy, head_width=0.4, head_length=0.6,
             fc='red', ec='red', alpha=0.8, linewidth=1.5)

# Mark start and end points
ax.plot(ref_x[0], ref_y[0], 'go', markersize=14, label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
ax.plot(ref_x[-1], ref_y[-1], 'ms', markersize=14, label='Ref End', zorder=5, markeredgecolor='purple', markeredgewidth=2)
ax.plot(odo_x[-1], odo_y[-1], 'c^', markersize=14, label='Odo End', zorder=5, markeredgecolor='darkblue', markeredgewidth=2)

# Labels and formatting
ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
ax.set_title('Robot Trajectory Comparison: Reference vs Odometry (Overlaid on Map)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.95)
ax.axis('equal')

# Add error statistics
final_error = np.sqrt((ref_x[-1] - odo_x[-1])**2 + (ref_y[-1] - odo_y[-1])**2)

# Calculate additional statistics
position_errors = np.sqrt((ref_x - odo_x)**2 + (ref_y - odo_y)**2)
mean_error = np.mean(position_errors)
max_error = np.max(position_errors)

# Create comprehensive statistics box
stats_text = (
    f'Final Position Error: {final_error:.3f} m\n'
    f'Mean Position Error: {mean_error:.3f} m\n'
    f'Max Position Error: {max_error:.3f} m'
)

ax.text(0.02, 0.98, stats_text, 
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='orange', linewidth=2))

plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Trajectories plotted and saved to 'trajectory_comparison.png'")
print(f"Reference trajectory: {len(ref_x)} points")
print(f"Odometry trajectory: {len(odo_x)} points")
print(f"Final position error: {final_error:.3f} m")
print(f"Mean position error: {mean_error:.3f} m")
print(f"Max position error: {max_error:.3f} m")
