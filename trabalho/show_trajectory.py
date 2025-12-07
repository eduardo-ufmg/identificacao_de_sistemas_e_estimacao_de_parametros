import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load map information
with open('map_info.json', 'r') as f:
    map_info = json.load(f)

# Load the map image
map_img = Image.open(map_info['image'])

# Load reference trajectory
ref_data = np.loadtxt('ref.csv', delimiter=',')
x_ref = ref_data[:, 0]
y_ref = ref_data[:, 1]
theta_ref = ref_data[:, 2]

# Create figure
fig, ax = plt.subplots()

# Display the map image with correct extent
extent = (map_info['xlimits'][0], map_info['xlimits'][1],
          map_info['ylimits'][0], map_info['ylimits'][1])
ax.imshow(map_img, extent=extent)

# Plot reference trajectory
ax.plot(x_ref, y_ref, label='Reference Trajectory')

# Mark initial position
ax.plot(x_ref[0], y_ref[0], 'o', label='Start')

# Mark final position
ax.plot(x_ref[-1], y_ref[-1], 'x', label='End')

# Set labels and title
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Reference Trajectory')
ax.legend(loc='best')

# Set axis limits according to map info
ax.set_xlim(map_info['xlimits'])
ax.set_ylim(map_info['ylimits'])

plt.tight_layout()
plt.show()
