"""
Extended Kalman Filter for Robot Localization

This implementation uses an EKF to estimate the robot's pose (x, y, theta)
by fusing differential odometry measurements and laser range measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
from scipy.ndimage import distance_transform_edt


class ExtendedKalmanFilter:
    def __init__(self, map_file='map.png', map_info_file='map_info.json'):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            map_file: Path to the map image
            map_info_file: Path to map metadata JSON
        """
        # State: [x, y, theta]
        self.state = np.zeros(3)
        self.covariance = np.eye(3)
        
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
        
        # Process noise covariance (odometry uncertainty)
        self.Q = np.diag([0.01, 0.01, 0.005])**2
        
        # Measurement noise covariance (laser uncertainty)
        self.R_laser = 0.05**2  # per laser reading
        
        # Use subset of laser beams for efficiency
        self.laser_indices = np.arange(0, self.num_laser_readings, 15)  # Every 15th beam
        
    def initialize(self, initial_state, initial_covariance):
        """
        Initialize the filter state.
        
        Args:
            initial_state: [x, y, theta]
            initial_covariance: 3x3 covariance matrix
        """
        self.state = np.array(initial_state)
        self.covariance = np.array(initial_covariance)
    
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
    
    def predict(self, odometry_delta):
        """
        EKF prediction step.
        
        Args:
            odometry_delta: [dx, dy, dtheta] incremental motion
        """
        dx, dy, dtheta = odometry_delta
        x, y, theta = self.state
        
        # State transition (motion model)
        # For simplicity, we assume dx, dy are in global frame
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Jacobian of motion model (identity for this simple model)
        F = np.eye(3)
        
        # Covariance prediction
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def compute_measurement_jacobian(self, laser_index):
        """
        Compute Jacobian of measurement function for a laser beam.
        
        Args:
            laser_index: Index of laser beam
            
        Returns:
            1x3 Jacobian matrix
        """
        x, y, theta = self.state
        angle = self.laser_angles[laser_index]
        beam_angle = theta + angle
        
        # Expected range
        expected_range = self.ray_cast(x, y, theta, angle)
        
        if expected_range >= self.max_laser_range:
            return np.zeros((1, 3))
        
        # Numerical differentiation for Jacobian
        epsilon = 1e-5
        H = np.zeros((1, 3))
        
        # dh/dx
        r_plus = self.ray_cast(x + epsilon, y, theta, angle)
        H[0, 0] = (r_plus - expected_range) / epsilon
        
        # dh/dy
        r_plus = self.ray_cast(x, y + epsilon, theta, angle)
        H[0, 1] = (r_plus - expected_range) / epsilon
        
        # dh/dtheta
        r_plus = self.ray_cast(x, y, theta + epsilon, angle)
        H[0, 2] = (r_plus - expected_range) / epsilon
        
        return H
    
    def update(self, laser_scan):
        """
        EKF update step using laser measurements.
        
        Args:
            laser_scan: Array of laser range measurements
        """
        x, y, theta = self.state
        
        # Process subset of laser measurements
        for idx in self.laser_indices:
            measured_range = laser_scan[idx]
            
            # Skip invalid measurements
            if measured_range >= self.max_laser_range:
                continue
            
            # Expected measurement
            expected_range = self.ray_cast(x, y, theta, self.laser_angles[idx])
            
            if expected_range >= self.max_laser_range:
                continue
            
            # Innovation
            innovation = measured_range - expected_range
            
            # Measurement Jacobian
            H = self.compute_measurement_jacobian(idx)
            
            if np.all(H == 0):
                continue
            
            # Innovation covariance
            S = H @ self.covariance @ H.T + self.R_laser
            
            # Kalman gain
            K = self.covariance @ H.T / S
            
            # State update
            self.state += (K * innovation).flatten()
            
            # Covariance update
            self.covariance = (np.eye(3) - K @ H) @ self.covariance
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
    
    @staticmethod
    def load_from_csv(filename):
        """Load data from CSV file."""
        return np.genfromtxt(filename, delimiter=',')
    
    def run_filter(self, odo_diff_file='odo_diff.csv', laser_file='laser.csv',
                   initial_state=None, initial_covariance=None):
        """
        Run the EKF on the entire dataset.
        
        Args:
            odo_diff_file: Path to differential odometry CSV
            laser_file: Path to laser scan CSV
            initial_state: Initial [x, y, theta] or None to use map_info
            initial_covariance: Initial 3x3 covariance or None for default
            
        Returns:
            Nx3 array of estimated poses [x, y, theta]
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
        
        print(f"Running Extended Kalman Filter...")
        
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
        
        print("Extended Kalman Filter complete!")
        return estimates, covariances


if __name__ == '__main__':
    # Run the EKF
    ekf = ExtendedKalmanFilter()
    
    # Get initial pose from map info
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_state = np.array(map_info['initial_pose'])
    
    # Run filter
    estimates, covariances = ekf.run_filter(
        initial_state=initial_state,
        initial_covariance=np.diag([0.1, 0.1, 0.05])**2
    )
    
    # Save results
    np.savetxt('ekf_estimates.csv', estimates, delimiter=',',
               header='x,y,theta', comments='')
    print("Saved estimates to 'ekf_estimates.csv'")
    
    # Load reference for comparison
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    # Compute errors
    position_errors = np.sqrt((ref_data[:, 0] - estimates[:, 0])**2 +
                             (ref_data[:, 1] - estimates[:, 1])**2)
    
    print(f"\nExtended Kalman Filter Results:")
    print(f"  Final error: {position_errors[-1]:.3f} m")
    print(f"  Mean error: {np.mean(position_errors):.3f} m")
    print(f"  Max error: {np.max(position_errors):.3f} m")
    
    # Print final covariance
    final_std = np.sqrt(np.diag(covariances[-1]))
    print(f"  Final uncertainty (std): x={final_std[0]:.3f}m, y={final_std[1]:.3f}m, theta={final_std[2]:.3f}rad")
