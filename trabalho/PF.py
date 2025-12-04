"""
Particle Filter for Robot Localization

This implementation uses a particle filter to estimate the robot's pose (x, y, theta)
by fusing differential odometry measurements and laser range measurements with a map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.ndimage import distance_transform_edt
import json
from OdometryDynamicModel import OdometryDynamicModel


class ParticleFilter:
    def __init__(self, num_particles=1000, map_file='map.png', map_info_file='map_info.json'):
        """
        Initialize the Particle Filter.
        
        Args:
            num_particles: Number of particles to use
            map_file: Path to the map image
            map_info_file: Path to map metadata JSON
        """
        self.num_particles = num_particles
        
        # Load map
        with open(map_info_file, 'r') as f:
            self.map_info = json.load(f)
        
        self.map_image = imread(map_file)
        # Convert to binary occupancy grid (1 = free, 0 = occupied)
        if len(self.map_image.shape) == 3:
            self.map_image = np.mean(self.map_image, axis=2)
        self.occupancy_grid = (self.map_image > 0.5).astype(float)
        
        # Compute distance transform for efficient likelihood calculation
        self.distance_transform = distance_transform_edt(self.occupancy_grid)
        
        # Map parameters
        self.resolution = self.map_info['resolution']
        self.xlimits = self.map_info['xlimits']
        self.ylimits = self.map_info['ylimits']
        self.nrow = self.map_info['nrow']
        self.ncol = self.map_info['ncol']
        
        # Laser parameters (361 readings, 0 to 360 degrees)
        self.num_laser_readings = 361
        self.laser_angles = np.linspace(0, 2*np.pi, self.num_laser_readings)
        self.max_laser_range = 8.183  # from the data
        
        # Particles: [x, y, theta]
        self.particles = np.zeros((num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles
        
        # Process noise (odometry uncertainty)
        self.motion_noise_std = np.array([0.05, 0.05, 0.02])  # [x, y, theta]
        
        # Measurement noise
        self.laser_noise_std = 0.1  # meters
        
        # Effective sample size threshold for resampling
        self.resample_threshold = 0.5
        
    def initialize_particles(self, initial_pose, initial_uncertainty):
        """
        Initialize particles around the initial pose.
        
        Args:
            initial_pose: (x, y, theta) initial estimate
            initial_uncertainty: std deviation for each dimension
        """
        for i in range(self.num_particles):
            self.particles[i] = initial_pose + np.random.randn(3) * initial_uncertainty
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def world_to_map(self, x, y):
        """Convert world coordinates to map grid indices."""
        col = int((x - self.xlimits[0]) / self.resolution)
        row = int((self.ylimits[1] - y) / self.resolution)
        return row, col
    
    def is_valid_position(self, x, y):
        """Check if a position is within map bounds and in free space."""
        row, col = self.world_to_map(x, y)
        if 0 <= row < self.nrow and 0 <= col < self.ncol:
            return self.occupancy_grid[row, col] > 0.5
        return False
    
    def predict(self, odometry_delta):
        """
        Prediction step: propagate particles using odometry.
        
        Args:
            odometry_delta: [dx, dy, dtheta] incremental motion
        """
        dx, dy, dtheta = odometry_delta
        
        # Add motion noise to each particle
        noise = np.random.randn(self.num_particles, 3) * self.motion_noise_std
        
        # Update particles
        self.particles[:, 0] += dx + noise[:, 0]
        self.particles[:, 1] += dy + noise[:, 1]
        self.particles[:, 2] += dtheta + noise[:, 2]
        
        # Normalize angles
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), 
                                          np.cos(self.particles[:, 2]))
    
    def ray_cast(self, x, y, theta, angle):
        """
        Simple ray casting to get expected laser range.
        
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
    
    def compute_likelihood(self, particle, laser_scan):
        """
        Compute likelihood of laser scan given particle pose.
        
        Args:
            particle: [x, y, theta]
            laser_scan: array of laser range measurements
            
        Returns:
            Likelihood value
        """
        x, y, theta = particle
        
        # Check if particle is in valid space
        if not self.is_valid_position(x, y):
            return 1e-10
        
        # Use subset of laser readings for efficiency (every 10th reading)
        indices = np.arange(0, self.num_laser_readings, 10)
        
        log_likelihood = 0.0
        for idx in indices:
            expected_range = self.ray_cast(x, y, theta, self.laser_angles[idx])
            measured_range = laser_scan[idx]
            
            # Skip invalid measurements
            if measured_range >= self.max_laser_range or expected_range >= self.max_laser_range:
                continue
            
            # Gaussian likelihood
            diff = measured_range - expected_range
            log_likelihood -= 0.5 * (diff / self.laser_noise_std) ** 2
        
        return np.exp(log_likelihood)
    
    def update(self, laser_scan):
        """
        Update step: compute weights based on laser measurements.
        
        Args:
            laser_scan: array of laser range measurements
        """
        # Compute likelihood for each particle
        for i in range(self.num_particles):
            self.weights[i] *= self.compute_likelihood(self.particles[i], laser_scan)
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All particles have zero weight, reinitialize uniformly
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """Resample particles based on weights (systematic resampling)."""
        # Compute effective sample size
        n_eff = 1.0 / np.sum(self.weights ** 2)
        
        # Only resample if effective sample size is too low
        if n_eff < self.resample_threshold * self.num_particles:
            # Systematic resampling
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0  # Ensure sum is exactly 1
            
            new_particles = np.zeros_like(self.particles)
            step = 1.0 / self.num_particles
            start = np.random.uniform(0, step)
            
            i = 0
            for j in range(self.num_particles):
                u = start + j * step
                while u > cumsum[i]:
                    i += 1
                new_particles[j] = self.particles[i]
            
            self.particles = new_particles
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate(self):
        """
        Get current state estimate (weighted mean of particles).
        
        Returns:
            [x, y, theta] state estimate
        """
        x = np.sum(self.weights * self.particles[:, 0])
        y = np.sum(self.weights * self.particles[:, 1])
        
        # For angle, use circular mean
        theta = np.arctan2(
            np.sum(self.weights * np.sin(self.particles[:, 2])),
            np.sum(self.weights * np.cos(self.particles[:, 2]))
        )
        
        return np.array([x, y, theta])
    
    @staticmethod
    def load_from_csv(filename):
        """Load data from CSV file."""
        return np.genfromtxt(filename, delimiter=',')
    
    def run_filter(self, odo_diff_file='odo_diff.csv', laser_file='laser.csv', 
                   initial_pose=None, initial_uncertainty=None):
        """
        Run the particle filter on the entire dataset.
        
        Args:
            odo_diff_file: Path to differential odometry CSV
            laser_file: Path to laser scan CSV
            initial_pose: Initial [x, y, theta] or None to use map_info
            initial_uncertainty: Std dev for initial particle distribution
            
        Returns:
            Nx3 array of estimated poses [x, y, theta]
        """
        # Load data
        odo_diff_data = self.load_from_csv(odo_diff_file)
        laser_data = self.load_from_csv(laser_file)
        
        n_steps = len(odo_diff_data)
        estimates = np.zeros((n_steps, 3))
        
        # Initialize
        if initial_pose is None:
            initial_pose = np.array(self.map_info['initial_pose'])
        if initial_uncertainty is None:
            initial_uncertainty = np.array([0.5, 0.5, 0.1])
        
        self.initialize_particles(initial_pose, initial_uncertainty)
        estimates[0] = initial_pose
        
        print(f"Running Particle Filter with {self.num_particles} particles...")
        
        # Process each timestep
        for t in range(1, n_steps):
            if t % 500 == 0:
                print(f"  Processing step {t}/{n_steps}")
            
            # Predict
            self.predict(odo_diff_data[t])
            
            # Update with laser measurement
            self.update(laser_data[t])
            
            # Resample
            self.resample()
            
            # Store estimate
            estimates[t] = self.estimate()
        
        print("Particle Filter complete!")
        return estimates


if __name__ == '__main__':
    # Run the particle filter
    pf = ParticleFilter(num_particles=500)
    
    # Get initial pose from map info
    with open('map_info.json', 'r') as f:
        map_info = json.load(f)
    initial_pose = np.array(map_info['initial_pose'])
    
    # Run filter
    estimates = pf.run_filter(initial_pose=initial_pose, 
                             initial_uncertainty=np.array([0.5, 0.5, 0.1]))
    
    # Save results
    np.savetxt('pf_estimates.csv', estimates, delimiter=',', 
               header='x,y,theta', comments='')
    print("Saved estimates to 'pf_estimates.csv'")
    
    # Load reference for comparison
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    
    # Compute errors
    position_errors = np.sqrt((ref_data[:, 0] - estimates[:, 0])**2 + 
                             (ref_data[:, 1] - estimates[:, 1])**2)
    
    print(f"\nParticle Filter Results:")
    print(f"  Final error: {position_errors[-1]:.3f} m")
    print(f"  Mean error: {np.mean(position_errors):.3f} m")
    print(f"  Max error: {np.max(position_errors):.3f} m")
