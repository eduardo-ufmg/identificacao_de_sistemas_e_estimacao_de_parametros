import numpy as np
import matplotlib.pyplot as plt
from OdometryModel import OdometryModel

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

# Plot reference trajectory
ax.plot(ref_x, ref_y, color='lime', label='Reference Trajectory')

# Plot odometry trajectory
ax.plot(odo_x, odo_y, color='red', label='Odometry Trajectory')

# Plot orientation arrows at regular intervals for reference
arrow_interval = max(1, len(ref_x) // 40)
for i in range(0, len(ref_x), arrow_interval):
    dx = np.cos(ref_angle[i]) * 0.5
    dy = np.sin(ref_angle[i]) * 0.5
    ax.arrow(ref_x[i], ref_y[i], dx, dy, head_width=0.5, head_length=0.75,
             fc='lime', ec='lime')

# Plot orientation arrows for odometry
for i in range(0, len(odo_x), arrow_interval):
    dx = np.cos(odo_angle[i]) * 0.5
    dy = np.sin(odo_angle[i]) * 0.5
    ax.arrow(odo_x[i], odo_y[i], dx, dy, head_width=0.5, head_length=0.75,
             fc='red', ec='red')

# Mark start and end points
ax.plot(ref_x[0], ref_y[0], 'go', markersize=12, label='Start', zorder=5)
ax.plot(ref_x[-1], ref_y[-1], 'mo', markersize=12, label='Ref End', zorder=5)
ax.plot(odo_x[-1], odo_y[-1], 'co', markersize=12, label='Odo End', zorder=5)

# Labels and formatting
ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_title('Robot Trajectory Comparison: Reference vs Odometry', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.axis('equal')

# Add error statistics
final_error = np.sqrt((ref_x[-1] - odo_x[-1])**2 + (ref_y[-1] - odo_y[-1])**2)
ax.text(0.02, 0.98, f'Final Position Error: {final_error:.2f} m', 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Trajectories plotted and saved to 'trajectory_comparison.png'")
print(f"Reference trajectory: {len(ref_x)} points")
print(f"Odometry trajectory: {len(odo_x)} points")
print(f"Final position error: {final_error:.3f} m")
