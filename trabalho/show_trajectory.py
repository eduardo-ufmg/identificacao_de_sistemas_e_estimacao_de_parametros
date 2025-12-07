import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot decimated reference trajectory with optional laser scan visualization"
)
parser.add_argument(
    "--laser", action="store_true", help="Show laser scan wavefront animation"
)
parser.add_argument(
    "--step", type=int, default=2, help="Step size for laser animation (default: 2)"
)
args = parser.parse_args()

# Load map information
with open("map_info.json", "r") as f:
    map_info = json.load(f)

# Load the map image
map_img = Image.open(map_info["image"])

# Load decimated reference trajectory
ref_data = np.loadtxt("ref_dec_trimmed.csv", delimiter=",")
x_ref = ref_data[:, 0]
y_ref = ref_data[:, 1]
theta_ref = ref_data[:, 2]

# Create figure
fig, ax = plt.subplots()

# Display the map image with correct extent
extent = (
    map_info["xlimits"][0],
    map_info["xlimits"][1],
    map_info["ylimits"][0],
    map_info["ylimits"][1],
)
ax.imshow(map_img, extent=extent)

# Plot reference trajectory
ax.plot(x_ref, y_ref, label="Reference Trajectory")

# Mark initial position
ax.plot(x_ref[0], y_ref[0], "o", label="Start")

# Mark final position
ax.plot(x_ref[-1], y_ref[-1], "x", label="End")

# Set labels and title
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Reference Trajectory" + (" with Laser Scans" if args.laser else ""))
ax.legend(loc="best")

# Set axis limits according to map info
ax.set_xlim(map_info["xlimits"])
ax.set_ylim(map_info["ylimits"])

if args.laser:
    # Load decimated laser data
    laser_data = np.loadtxt("laser_dec_trimmed.csv", delimiter=",")

    # Laser scanner parameters (typical for a 360-degree scanner)
    num_beams = laser_data.shape[1]
    angle_min = -np.pi  # -180 degrees
    angle_max = np.pi  # 180 degrees
    angles = np.linspace(angle_min, angle_max, num_beams)

    # Initialize laser scan plot
    (laser_points,) = ax.plot([], [], "r.", label="Laser Scan")
    (robot_pos,) = ax.plot([], [], "yo", label="Robot Position")

    # Set title for animation
    ax.set_title(f"Reference Trajectory with Laser Scans")

    # Update legend
    ax.legend(loc="best")

    def init():
        laser_points.set_data([], [])
        robot_pos.set_data([], [])
        return laser_points, robot_pos

    def update(frame):
        idx = frame * args.step
        if idx >= len(x_ref):
            idx = len(x_ref) - 1

        # Current robot pose
        x, y, theta = x_ref[idx], y_ref[idx], theta_ref[idx]

        # Get laser ranges for this timestep
        ranges = laser_data[idx, :]

        # Convert laser ranges to Cartesian coordinates
        # Laser angles are relative to robot heading
        laser_x = []
        laser_y = []
        for i, (angle, r) in enumerate(zip(angles, ranges)):
            if r > 0.1 and r < 8.0:  # Filter out invalid readings
                # Transform to global coordinates
                global_angle = theta + angle
                laser_x.append(x + r * np.cos(global_angle))
                laser_y.append(y + r * np.sin(global_angle))

        laser_points.set_data(laser_x, laser_y)
        robot_pos.set_data([x], [y])

        return laser_points, robot_pos

    frames = max(1, math.ceil(len(x_ref) / max(1, args.step)))
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000,
        blit=True,
        repeat=True,
    )

plt.show()
