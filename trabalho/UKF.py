import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
	from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
except ImportError as exc:
	raise SystemExit('filterpy is required for UKF; please install it (pip install filterpy)') from exc

from LaserDynamicModel import LaserDynamicModel
from OdometryDynamicModel import OdometryDynamicModel

class UKFEstimator:
	"""Minimal UKF estimator that fuses odometry deltas (control) and pose measurements."""

	def __init__(self, initial_state: np.ndarray, q_std=(0.02, 0.02, 0.05), r_std=(0.05, 0.05, 0.1)):
		self.initial_state = initial_state.copy()
		self.q_std = np.array(q_std, dtype=float)
		self.r_std = np.array(r_std, dtype=float)

		n_dim = 3
		points = MerweScaledSigmaPoints(n=n_dim, alpha=0.1, beta=2.0, kappa=0.0)

		def fx(x, dt, u):
			return x + u  # odometry deltas already expressed in global frame

		def hx(x):
			return x  # measurements are pose estimates

		self.ukf = UnscentedKalmanFilter(dim_x=n_dim, dim_z=n_dim, fx=fx, hx=hx, dt=1.0, points=points)
		self.ukf.x = self.initial_state.copy()
		self.ukf.P = np.diag([0.1, 0.1, 0.2])
		self.ukf.Q = np.diag(np.square(self.q_std))
		self.ukf.R = np.diag(np.square(self.r_std))

	def step(self, control_delta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
		"""Run one UKF predict/update cycle and return the posterior state."""
		self.ukf.predict(u=control_delta)
		self.ukf.update(measurement)
		return self.ukf.x.copy()

	def run(self, controls: np.ndarray, measurements: np.ndarray) -> np.ndarray:
		"""Run UKF over full sequences; returns states including initial."""
		states = [self.ukf.x.copy()]
		for u, z in zip(controls, measurements):
			states.append(self.step(u, z))
		return np.array(states)


def load_model(model_json_path: str):
	with open(model_json_path, 'r') as f:
		data = json.load(f)
	A = np.array(data['A'])
	B = np.array(data['B'])
	bias = np.array(data['bias'])
	return A, B, bias


def build_laser_measurements(laser_path: str, model_path: str, initial_pose: np.ndarray):
	A, B, bias = load_model(model_path)
	laser_data = np.loadtxt(laser_path, delimiter=',')
	if laser_data.ndim == 1:
		laser_data = laser_data.reshape(1, -1)
	model = LaserDynamicModel(A, B, bias, initial_pose)
	return model.simulate(laser_data)


def build_odometry_trajectory(odo_diff_path: str, initial_pose: np.ndarray):
    odo_diff = np.loadtxt(odo_diff_path, delimiter=',')
    if odo_diff.ndim == 1:
        odo_diff = odo_diff.reshape(-1, 3)
    model = OdometryDynamicModel(initial_pose)
    return model.simulate(odo_diff)


def load_tuned_params(tuned_path: str):
	"""Load tuned parameters from JSON if available; return (q_std, r_std) or None."""
	try:
		with open(tuned_path, 'r') as f:
			data = json.load(f)
		q_std = tuple(data['q_std'])
		r_std = tuple(data['r_std'])
		return q_std, r_std
	except (FileNotFoundError, json.JSONDecodeError, KeyError):
		return None


def parse_args():
	parser = argparse.ArgumentParser(description='UKF fusion of odometry deltas and laser-based dynamic model')
	parser.add_argument('--odo-diff', default='odo_dec_diff.csv', help='Path to odometry differences CSV (dx, dy, dtheta)')
	parser.add_argument('--laser', default='laser_dec.csv', help='Path to laser data CSV')
	parser.add_argument('--model', default='laser_model.json', help='Path to laser model JSON')
	parser.add_argument('--map-info', default='map_info.json', help='Path to map info JSON')
	parser.add_argument('--tuned', default='ukf_tuned.json', help='Path to tuned parameters JSON (optional)')
	parser.add_argument('--q-std', type=float, nargs=3, default=None, help='Process noise std for [x, y, theta] (overrides tuned if set)')
	parser.add_argument('--r-std', type=float, nargs=3, default=None, help='Measurement noise std for [x, y, theta] (overrides tuned if set)')
	return parser.parse_args()


def main():
	args = parse_args()

	# Determine Q and R std: prefer CLI args, then tuned, then defaults
	q_std = args.q_std
	r_std = args.r_std

	if q_std is None or r_std is None:
		tuned = load_tuned_params(args.tuned)
		if tuned:
			tuned_q, tuned_r = tuned
			if q_std is None:
				q_std = tuned_q
			if r_std is None:
				r_std = tuned_r
			print(f'Loaded tuned parameters from {args.tuned}')
		else:
			if q_std is None:
				q_std = (0.02, 0.02, 0.05)
			if r_std is None:
				r_std = (0.05, 0.05, 0.1)
			print('Using default parameters')

	# Map and initial pose
	with open(args.map_info, 'r') as f:
		map_info = json.load(f)
	map_img = Image.open(map_info['image'])
	initial_pose = np.array(map_info['initial_pose'], dtype=float)

	# Data
	odom_traj = build_odometry_trajectory(args.odo_diff, initial_pose)
	laser_traj = build_laser_measurements(args.laser, args.model, initial_pose)

	# UKF Estimation
	ukf = UKFEstimator(initial_pose, q_std=q_std, r_std=r_std)
	est_states = ukf.run(
        controls=odom_traj[1:] - odom_traj[:-1],
        measurements=laser_traj[1:],
    )
	

	# Plot
	fig, ax = plt.subplots(figsize=(12, 10))
	extent = (
		map_info['xlimits'][0],
		map_info['xlimits'][1],
		map_info['ylimits'][0],
		map_info['ylimits'][1],
	)
	ax.imshow(map_img, extent=extent)

	ax.plot(odom_traj[:, 0], odom_traj[:, 1], 'b-', label='Odometry only')
	ax.plot(laser_traj[:, 0], laser_traj[:, 1], 'r-', label='Laser model')
	ax.plot(est_states[:, 0], est_states[:, 1], color='lime', linestyle='-', label='UKF estimate')

	ax.plot(est_states[0, 0], est_states[0, 1], 'o', label='Start')
	ax.plot(est_states[-1, 0], est_states[-1, 1], 'x', label='End')

	ax.set_xlabel('X Position (m)')
	ax.set_ylabel('Y Position (m)')
	ax.set_title('UKF Fusion: Odometry + Laser Dynamic Model')
	ax.set_xlim(map_info['xlimits'])
	ax.set_ylim(map_info['ylimits'])
	ax.legend(loc='best')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()