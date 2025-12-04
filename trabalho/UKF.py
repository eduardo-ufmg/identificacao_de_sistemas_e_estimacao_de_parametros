"""
Unscented Kalman Filter for Robot Localization

This implementation uses a UKF to estimate the robot's pose (x, y, theta)
by fusing differential odometry measurements and laser range measurements.
The UKF handles nonlinearities better than EKF by using the unscented transform.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
from scipy.ndimage import distance_transform_edt


class UnscentedKalmanFilter:
    def __init__(self, map_file='map.png', map_info_file='map_info.json'):
        """
        Initialize the Unscented Kalman Filter.
        
        Args:
            map_file: Path to the map image
            map_info_file: Path to map metadata JSON
        """
        # State: [x, y, theta]
        self.n_state = 3
        self.state = np.zeros(self.n_state)
        self.covariance = np.eye(self.n_state)
        
        # UKF parameters
        self.alpha = 1e-3  # Spread of sigma points
        self.beta = 2.0    # Incorporates prior knowledge (2 is optimal for Gaussian)
        self.kappa = 0.0   # Secondary scaling parameter
        
        self.lambda_ = self.alpha**2 * (self.n_state + self.kappa) - self.n_state
        self.n_sigma = 2 * self.n_state + 1
        
        # Weights for sigma points
        self.weights_mean = np.zeros(self.n_sigma)
        self.weights_cov = np.zeros(self.n_sigma)
        
        self.weights_mean[0] = self.lambda_ / (self.n_state + self.lambda_)
        self.weights_cov[0] = self.lambda_ / (self.n_state + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, self.n_sigma):
            self.weights_mean[i] = 1.0 / (2 * (self.n_state + self.lambda_))
            self.weights_cov[i] = 1.0 / (2 * (self.n_state + self.lambda_))
        
        # Load map
        with open(map_info_file, 'r') as f:
            self.map_info = json.load(f)
        
        self.map_image = imread(map_file)
        # Convert to binary occupancy grid
        if len(self.map_image.shape) == 3:
            self.map_image = np.mean(self.map_image, axis=2)
        self.occupancy_grid = (self.map_image > 0.5).astype(float)
        
        # Compute distance transform
        self.distance_transform = distance_transform_edt(self.occupancy_grid)
        
        # Map parameters
        self.resolution = self.map_info['resolution']
        self.xlimits = self.map_info['xlimits']
        self.ylimits = self.map_info['ylimits']
        self.nrow = self.map_info['nrow']
        self.ncol = self.map_info['ncol']
        
        # Laser parameters
        self.num_laser_readings = 361
        self.laser_angles = np.linspace(0, 2*np.pi, self.num_laser_readings)
        self.max_laser_range = 8.183
        
        # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.005])**2
        
        # Measurement noise covariance
        self.R_laser = 0.05**2
        
        # Use subset of laser beams
        self.laser_indices = np.arange(0, self.num_laser_readings, 15)
        
    def initialize(self, initial_state, initial_covariance):
        """
        Initialize the filter state.
        
        Args:
            initial_state: [x, y, theta]
            initial_covariance: 3x3 covariance matrix
        """
        self.state = np.array(initial_state)
        self.covariance = np.array(initial_covariance)
    
    def generate_sigma_points(self, mean, covariance):
        """
        Generate sigma points using the unscented transform.
        
        Args:
            mean: State mean vector
            covariance: State covariance matrix
            
        Returns:
            Array of sigma points (n_sigma x n_state)
        """
        sigma_points = np.zeros((self.n_sigma, self.n_state))
        sigma_points[0] = mean
        
        # Matrix square root using Cholesky decomposition
        try:
            L = np.linalg.cholesky((self.n_state + self.lambda_) * covariance)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            L = eigvecs @ np.diag(np.sqrt((self.n_state + self.lambda_) * eigvals))
        
        for i in range(self.n_state):
            sigma_points[i + 1] = mean + L[:, i]
            sigma_points[i + 1 + self.n_state] = mean - L[:, i]
        
        return sigma_points
    
    def world_to_map(self, x, y):
        """Convert world coordinates to map grid indices."""
        col = int((x - self.xlimits[0]) / self.resolution)
        row = int((self.ylimits[1] - y) / self.resolution)
        return row, col
    
    def ray_cast(self, x, y, theta, angle):
        """
        Ray casting to get expected laser range.
        
        Args:
            x, y: robot position
            theta: robot orientation
            angle: laser beam angle relative to robot
            
        Returns:
            Expected range measurement
        """
        beam_angle = theta + angle
        max_range = self.max_laser_range
        step_size = self.resolution
        
        for r in np.arange(step_size, max_range, step_size):
            px = x + r * np.cos(beam_angle)
            py = y + r * np.sin(beam_angle)
            
            row, col = self.world_to_map(px, py)
            if not (0 <= row < self.nrow and 0 <= col < self.ncol):
                return max_range
            if self.occupancy_grid[row, col] < 0.5:
                return r
        
        return max_range
    
    def motion_model(self, state, odometry_delta):
        """
        Apply motion model to a state.
        
        Args:
            state: [x, y, theta]
            odometry_delta: [dx, dy, dtheta]
            
        Returns:
            Updated state
        """
        new_state = state.copy()
        new_state[0] += odometry_delta[0]
        new_state[1] += odometry_delta[1]
        new_state[2] += odometry_delta[2]
        
        # Normalize angle
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        
        return new_state
    
    def measurement_model(self, state, laser_index):
        """
        Predict laser measurement for a given state.
        
        Args:
            state: [x, y, theta]
            laser_index: Index of laser beam
            
        Returns:
            Expected range measurement
        """
        x, y, theta = state
        angle = self.laser_angles[laser_index]
        return self.ray_cast(x, y, theta, angle)
    
    def predict(self, odometry_delta):
        """
        UKF prediction step.
        
        Args:
            odometry_delta: [dx, dy, dtheta] incremental motion
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.state, self.covariance)
        
        # Propagate sigma points through motion model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i] = self.motion_model(sigma_points[i], odometry_delta)
        
        # Compute predicted mean
        self.state = np.sum(self.weights_mean[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Normalize angle in mean
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Compute predicted covariance
        self.covariance = np.zeros((self.n_state, self.n_state))
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i] - self.state
            # Handle angle wraparound
            diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))
            self.covariance += self.weights_cov[i] * np.outer(diff, diff)
        
        self.covariance += self.Q
    
    def update(self, laser_scan):
        """
        UKF update step using laser measurements.
        
        Args:
            laser_scan: Array of laser range measurements
        """
        # Process each laser measurement
        for idx in self.laser_indices:
            measured_range = laser_scan[idx]
            
            # Skip invalid measurements
            if measured_range >= self.max_laser_range:
                continue
            
            # Generate sigma points from current state
            sigma_points = self.generate_sigma_points(self.state, self.covariance)
            
            # Predict measurements for each sigma point
            z_sigma = np.zeros(self.n_sigma)
            for i in range(self.n_sigma):
                z_sigma[i] = self.measurement_model(sigma_points[i], idx)
            
            # Skip if all predictions are at max range
            if np.all(z_sigma >= self.max_laser_range):
                continue
            
            # Predicted measurement mean
            z_pred = np.sum(self.weights_mean * z_sigma)
            
            # Innovation covariance
            S = self.R_laser
            for i in range(self.n_sigma):
                diff_z = z_sigma[i] - z_pred
                S += self.weights_cov[i] * diff_z * diff_z
            
            # Cross-covariance
            C = np.zeros(self.n_state)
            for i in range(self.n_sigma):
                diff_x = sigma_points[i] - self.state
                diff_x[2] = np.arctan2(np.sin(diff_x[2]), np.cos(diff_x[2]))
                diff_z = z_sigma[i] - z_pred
                C += self.weights_cov[i] * diff_x * diff_z
            
            # Kalman gain
            K = C / S
            
            # Innovation
            innovation = measured_range - z_pred
            
            # State update
            self.state += K * innovation
            
            # Normalize angle
            self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
            
            # Covariance update
            self.covariance -= np.outer(K, K) * S
    
    @staticmethod
    def load_from_csv(filename):
        """Load data from CSV file."""
        return np.genfromtxt(filename, delimiter=',')
    
    def run_filter(self, odo_diff_file='odo_diff.csv', laser_file='laser.csv',
                   initial_state=None, initial_covariance=None):
        """
        Run the UKF on the entire dataset.
        
        Args:
            odo_diff_file: Path to differential odometry CSV
            laser_file: Path to laser scan CSV
            initial_state: Initial [x, y, theta] or None to use map_info
            initial_covariance: Initial 3x3 covariance or None for default
            
        Returns:
            Tuple of (estimates, covariances)
            - estimates: Nx3 array of estimated poses [x, y, theta]
            - covariances: Nx3x3 array of covariance matrices
        """
        # Load data
        odo_diff_data = self.load_from_csv(odo_diff_file)
        laser_data = self.load_from_csv(laser_file)
        
        n_steps = len(odo_diff_data)
        estimates = np.zeros((n_steps, 3))
        covariances = np.zeros((n_steps, 3, 3))
        
        # Initialize
        if initial_state is None:
            initial_state = np.array(self.map_info['initial_pose'])
        if initial_covariance is None:
            initial_covariance = np.diag([0.1, 0.1, 0.05])**2
        
        self.initialize(initial_state, initial_covariance)
        estimates[0] = initial_state
        covariances[0] = initial_covariance
        
        print(f"Running Unscented Kalman Filter...")
        
        # Process each timestep
        for t in range(1, n_steps):
            if t % 500 == 0:
                print(f"  Processing step {t}/{n_steps}")
            
            # Predict
            self.predict(odo_diff_data[t])
            
            # Update with laser measurement
            self.update(laser_data[t])
            
            # Store estimate
            estimates[t] = self.state.copy()
            covariances[t] = self.covariance.copy()
        
        print("Unscented Kalman Filter complete!")
        return estimates, covariances


if __name__ == '__main__':
    # Run the UKF
    ukf = UnscentedKalmanFilter()
    
    # Get initial pose from map info
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_state = np.array(map_info['initial_pose'])
    
    # Run filter
    estimates, covariances = ukf.run_filter(
        initial_state=initial_state,
        initial_covariance=np.diag([0.1, 0.1, 0.05])**2
    )
    
    # Save results
    np.savetxt('ukf_estimates.csv', estimates, delimiter=',',
               header='x,y,theta', comments='')
    print("Saved estimates to 'ukf_estimates.csv'")
    
    # Load reference for comparison
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    # Compute errors
    position_errors = np.sqrt((ref_data[:, 0] - estimates[:, 0])**2 +
                             (ref_data[:, 1] - estimates[:, 1])**2)
    
    print(f"\nUnscented Kalman Filter Results:")
    print(f"  Final error: {position_errors[-1]:.3f} m")
    print(f"  Mean error: {np.mean(position_errors):.3f} m")
    print(f"  Max error: {np.max(position_errors):.3f} m")
    
    # Print final covariance
    final_std = np.sqrt(np.diag(covariances[-1]))
    print(f"  Final uncertainty (std): x={final_std[0]:.3f}m, y={final_std[1]:.3f}m, theta={final_std[2]:.3f}rad")
