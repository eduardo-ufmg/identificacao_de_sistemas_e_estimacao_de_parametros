import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class ParticleFilter:
    """
    Particle Filter for robot localization using odometry and laser range measurements.
    
    State vector: [x, y, theta] - robot position and orientation
    Uses particle representation to handle highly nonlinear models and non-Gaussian distributions.
    """
    
    def __init__(self, initial_pose=(0, 0, 0), n_particles=1000, 
                 process_noise=None, measurement_noise=None):
        """
        Initialize the Particle Filter.
        
        Args:
            initial_pose: Initial robot pose (x, y, theta)
            n_particles: Number of particles
            process_noise: Process noise covariance matrix Q (3x3)
            measurement_noise: Measurement noise variance R
        """
        self.n_particles = n_particles
        
        # Initialize particles around initial pose
        self.particles = np.zeros((n_particles, 3))
        initial_std = [0.1, 0.1, 0.05]  # Initial uncertainty
        
        for i in range(3):
            self.particles[:, i] = np.random.normal(
                initial_pose[i], initial_std[i], n_particles
            )
        
        # Initialize weights (uniform)
        self.weights = np.ones(n_particles) / n_particles
        
        # Process noise covariance
        if process_noise is None:
            self.Q = np.diag([0.1, 0.1, 0.05])  # [x, y, theta] process noise
        else:
            self.Q = process_noise
            
        # Measurement noise variance
        if measurement_noise is None:
            self.measurement_noise = 0.1
        else:
            self.measurement_noise = measurement_noise
            
        # Laser scanner configuration
        self.laser_angles = np.linspace(-np.pi/2, np.pi/2, 361)  # -90° to +90°
        self.max_range = 8.183  # Maximum laser range from data
        
        # Simple landmark map (walls/obstacles for range measurements)
        self.landmarks = self._create_simple_map()
        
        # Resampling threshold
        self.resample_threshold = 0.5
        
    def _create_simple_map(self):
        """
        Create a simple map with walls for laser range calculations.
        """
        landmarks = []
        
        # Room boundaries (approximate from laser data analysis)
        wall_points = [
            # Front wall points
            [(i*0.1, 8.0) for i in range(-50, 51)],
            # Left wall points  
            [(-5.0, i*0.1) for i in range(0, 80)],
            # Right wall points
            [(5.0, i*0.1) for i in range(0, 80)],
        ]
        
        for wall in wall_points:
            landmarks.extend(wall)
            
        return np.array(landmarks)
    
    def _motion_model(self, particle, odometry):
        """
        Apply motion model with noise to a particle.
        
        Args:
            particle: Current particle state [x, y, theta]
            odometry: Odometry input [dx, dy, dtheta]
            
        Returns:
            new_particle: Updated particle state
        """
        dx, dy, dtheta = odometry
        
        # Add process noise
        noise = np.random.multivariate_normal([0, 0, 0], self.Q)
        
        cos_theta = np.cos(particle[2])
        sin_theta = np.sin(particle[2])
        
        new_particle = np.array([
            particle[0] + dx * cos_theta - dy * sin_theta + noise[0],
            particle[1] + dx * sin_theta + dy * cos_theta + noise[1],
            particle[2] + dtheta + noise[2]
        ])
        
        # Wrap angle to [-pi, pi]
        new_particle[2] = np.arctan2(np.sin(new_particle[2]), np.cos(new_particle[2]))
        
        return new_particle
    
    def _measurement_likelihood(self, particle, laser_measurements):
        """
        Calculate likelihood of measurements given particle pose.
        
        Args:
            particle: Particle state [x, y, theta]
            laser_measurements: Array of laser range measurements
            
        Returns:
            likelihood: Likelihood of measurements
        """
        x, y, theta = particle
        
        # Use subset of measurements for efficiency
        measurement_indices = range(0, len(laser_measurements), 20)  # Every 20th beam
        
        log_likelihood = 0.0
        n_valid = 0
        
        for i in measurement_indices:
            measured_range = laser_measurements[i]
            
            if measured_range < self.max_range:  # Valid measurement
                angle = self.laser_angles[i]
                global_angle = theta + angle
                
                # Expected measurement using ray casting
                expected_range = self._ray_cast(x, y, global_angle)
                
                if expected_range < self.max_range:
                    # Gaussian likelihood
                    diff = measured_range - expected_range
                    log_likelihood -= 0.5 * (diff**2) / self.measurement_noise
                    n_valid += 1
        
        if n_valid == 0:
            return 1e-10  # Very small likelihood if no valid measurements
        
        # Normalize by number of measurements
        log_likelihood /= n_valid
        
        return np.exp(log_likelihood)
    
    def _ray_cast(self, x, y, angle):
        """
        Simple ray casting to find expected range measurement.
        """
        step_size = 0.1
        max_steps = int(self.max_range / step_size)
        
        for step in range(1, max_steps):
            test_x = x + step * step_size * np.cos(angle)
            test_y = y + step * step_size * np.sin(angle)
            
            # Check if ray hits any landmark
            distances = np.sqrt((self.landmarks[:, 0] - test_x)**2 + 
                              (self.landmarks[:, 1] - test_y)**2)
            
            if np.min(distances) < 0.2:  # Hit threshold
                return step * step_size
                
        return self.max_range
    
    def predict(self, odometry):
        """
        Prediction step: propagate particles through motion model.
        
        Args:
            odometry: [dx, dy, dtheta] odometry increments
        """
        for i in range(self.n_particles):
            self.particles[i] = self._motion_model(self.particles[i], odometry)
    
    def update(self, laser_measurements):
        """
        Update step: weight particles based on measurement likelihood.
        
        Args:
            laser_measurements: Array of 361 laser range measurements
        """
        # Update weights based on measurement likelihood
        for i in range(self.n_particles):
            self.weights[i] *= self._measurement_likelihood(self.particles[i], laser_measurements)
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Check if resampling is needed
        if self._effective_sample_size() < self.resample_threshold * self.n_particles:
            self._resample()
    
    def _effective_sample_size(self):
        """Calculate effective sample size."""
        return 1.0 / np.sum(self.weights**2)
    
    def _resample(self):
        """
        Resample particles based on their weights (systematic resampling).
        """
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly 1
        
        # Systematic resampling
        step = 1.0 / self.n_particles
        r = np.random.uniform(0, step)
        
        new_particles = np.zeros_like(self.particles)
        i = 0
        
        for j in range(self.n_particles):
            u = r + j * step
            while u > cumulative_sum[i]:
                i += 1
            new_particles[j] = self.particles[i]
        
        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_state(self):
        """Return weighted average state estimate."""
        # Weighted mean for x and y
        x_est = np.sum(self.weights * self.particles[:, 0])
        y_est = np.sum(self.weights * self.particles[:, 1])
        
        # Circular mean for angle
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta_est = np.arctan2(sin_sum, cos_sum)
        
        return np.array([x_est, y_est, theta_est])
    
    def get_particles(self):
        """Return current particles and weights."""
        return self.particles.copy(), self.weights.copy()

def run_pf_estimation(initial_pose, odometry_data, laser_data, n_particles=1000):
    """
    Run Particle Filter estimation on the provided data.
    
    Args:
        initial_pose: Initial robot pose (x, y, theta)
        odometry_data: Nx3 array of odometry increments
        laser_data: Nx361 array of laser measurements
        n_particles: Number of particles
    
    Returns:
        Nx3 array of estimated poses
    """
    pf = ParticleFilter(initial_pose, n_particles)
    
    n_steps = len(odometry_data)
    estimated_poses = np.zeros((n_steps, 3))
    
    # Initialize with first pose
    estimated_poses[0] = initial_pose
    
    # Process each time step
    for i in range(1, n_steps):
        # Predict step
        pf.predict(odometry_data[i] - odometry_data[i-1])
        
        # Update step with laser measurements
        pf.update(laser_data[i])
        
        # Store estimate
        estimated_poses[i] = pf.get_state()
        
        if i % 500 == 0:
            print(f"PF: Processed {i}/{n_steps} measurements")
    
    return estimated_poses

def visualize_particles(pf, ref_pose, step):
    """
    Visualize particles for debugging/demonstration.
    """
    particles, weights = pf.get_particles()
    
    plt.figure(figsize=(10, 8))
    
    # Plot particles with size proportional to weight
    sizes = weights * 1000 + 1  # Scale for visibility
    plt.scatter(particles[:, 0], particles[:, 1], s=sizes, alpha=0.6, c='blue')
    
    # Plot estimate
    estimate = pf.get_state()
    plt.plot(estimate[0], estimate[1], 'ro', markersize=10, label='PF Estimate')
    
    # Plot reference
    plt.plot(ref_pose[0], ref_pose[1], 'go', markersize=10, label='Reference')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Particle Filter - Step {step}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig(f'pf_particles_step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load data
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    odo_data = np.genfromtxt('odo.csv', delimiter=',')
    laser_data = np.genfromtxt('laser.csv', delimiter=',')
    
    # Initial pose from reference
    initial_pose = (ref_data[0, 0], ref_data[0, 1], ref_data[0, 2])
    
    # Run Particle Filter estimation
    print("Running Particle Filter...")
    pf_poses = run_pf_estimation(initial_pose, odo_data, laser_data, n_particles=2000)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    plt.plot(ref_data[:, 0], ref_data[:, 1], 'g-', label='Reference', linewidth=2)
    plt.plot(pf_poses[:, 0], pf_poses[:, 1], 'm-', label='PF Estimate', linewidth=2)
    
    # Mark start and end
    plt.plot(ref_data[0, 0], ref_data[0, 1], 'go', markersize=10, label='Start')
    plt.plot(ref_data[-1, 0], ref_data[-1, 1], 'ro', markersize=10, label='Reference End')
    plt.plot(pf_poses[-1, 0], pf_poses[-1, 1], 'bo', markersize=10, label='PF End')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Particle Filter Localization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Calculate and display final error
    final_error = np.sqrt((ref_data[-1, 0] - pf_poses[-1, 0])**2 + 
                         (ref_data[-1, 1] - pf_poses[-1, 1])**2)
    plt.text(0.02, 0.98, f'Final Position Error: {final_error:.2f} m', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('pf_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"PF Final position error: {final_error:.3f} m")
