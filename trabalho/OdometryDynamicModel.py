import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class OdometryDynamicModel:
	"""
	Simulates robot position from odometry deltas in the global frame.
	Assumes odo_diff data contains [dx, dy, dtheta] increments already in global coords.
	"""

	def __init__(self, initial_pose: np.ndarray):
		"""
		Initialize the model with initial state.
		
		Args:
			initial_pose: array [x, y, theta]
		"""
		self.state = initial_pose.copy()

	def step(self, delta: np.ndarray) -> np.ndarray:
		"""
		Apply one step of odometry delta to the state.
		
		Args:
			delta: array [dx, dy, dtheta] in global frame
		
		Returns:
			Updated state [x, y, theta]
		"""
		self.state[0] += delta[0]
		self.state[1] += delta[1]
		self.state[2] += delta[2]
		return self.state.copy()

	def simulate(self, odo_diff: np.ndarray) -> np.ndarray:
		"""
		Simulate the full trajectory from odometry deltas.
		
		Args:
			odo_diff: shape (n_steps, 3) with [dx, dy, dtheta] per step
		
		Returns:
			Trajectory array of shape (n_steps + 1, 3) including initial pose
		"""
		trajectory = [self.state.copy()]
		for delta in odo_diff:
			trajectory.append(self.step(delta))
		return np.array(trajectory)


def main():
	parser = argparse.ArgumentParser(description='Simulate robot position from odometry deltas and plot on map')
	parser.add_argument('--odo-diff', default='odo_dec_diff.csv', help='Path to odometry differences CSV (dx, dy, dtheta)')
	parser.add_argument('--map-info', default='map_info.json', help='Path to map info JSON')
	args = parser.parse_args()

	# Load map info and image
	with open(args.map_info, 'r') as f:
		map_info = json.load(f)

	map_img = Image.open(map_info['image'])
	initial_pose = np.array(map_info['initial_pose'])

	# Load odometry deltas
	odo_diff = np.loadtxt(args.odo_diff, delimiter=',')
	if odo_diff.ndim == 1:
		odo_diff = odo_diff.reshape(-1, 1)

	# Simulate trajectory
	model = OdometryDynamicModel(initial_pose)
	trajectory = model.simulate(odo_diff)

	x_traj = trajectory[:, 0]
	y_traj = trajectory[:, 1]

	# Plot
	fig, ax = plt.subplots(figsize=(12, 10))

	extent = (
		map_info['xlimits'][0],
		map_info['xlimits'][1],
		map_info['ylimits'][0],
		map_info['ylimits'][1],
	)
	ax.imshow(map_img, extent=extent)

	ax.plot(x_traj, y_traj, '.-', label='Odometry Trajectory')
	ax.plot(x_traj[0], y_traj[0], 'o', label='Start')
	ax.plot(x_traj[-1], y_traj[-1], 'x', label='End')

	ax.set_xlabel('X Position (m)')
	ax.set_ylabel('Y Position (m)')
	ax.set_title('Robot Trajectory from Odometry')
	ax.set_xlim(map_info['xlimits'])
	ax.set_ylim(map_info['ylimits'])
	ax.legend(loc='best')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()
