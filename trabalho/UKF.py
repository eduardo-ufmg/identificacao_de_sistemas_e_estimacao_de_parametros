import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for robot localization using odometry and laser range measurements.
    
    State vector: [x, y, theta] - robot position and orientation
    Uses sigma points to handle nonlinearities in motion and measurement models.
    """
    
    def __init__(self, initial_pose=(0, 0, 0), process_noise=None, measurement_noise=None):
        """
        Initialize the UKF.
        
        Args:
            initial_pose: Initial robot pose (x, y, theta)
            process_noise: Process noise covariance matrix Q (3x3)
            measurement_noise: Measurement noise variance R
        """
        # State dimension
        self.n = 3  # [x, y, theta]
        
        # State vector [x, y, theta]
        self.x = np.array(initial_pose, dtype=float)
        
        # State covariance matrix
        self.P = np.eye(3) * 0.1
        
        # Process noise covariance
        if process_noise is None:
            self.Q = np.diag([0.1, 0.1, 0.05])  # [x, y, theta] process noise
        else:
            self.Q = process_noise
            
        # Measurement noise covariance
        if measurement_noise is None:
            self.R = 0.1  # Laser measurement noise variance
        else:
            self.R = measurement_noise
            
        # UKF parameters
        self.alpha = 1e-3    # Spread of sigma points
        self.beta = 2.0      # Prior knowledge of distribution (2 is optimal for Gaussian)
        self.kappa = 0       # Secondary scaling parameter
        
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weights for sigma points
        self.Wm = np.zeros(2 * self.n + 1)  # Weights for means
        self.Wc = np.zeros(2 * self.n + 1)  # Weights for covariances
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lambda_))
            self.Wc[i] = self.Wm[i]
        
        # Laser scanner configuration
        self.laser_angles = np.linspace(-np.pi/2, np.pi/2, 361)  # -90° to +90°
        self.max_range = 8.183  # Maximum laser range from data
        
        # Simple landmark map (walls/obstacles for range measurements)
        self.landmarks = self._create_simple_map()
        
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
    
    def _generate_sigma_points(self, x, P):
        """
        Generate sigma points for the unscented transform.
        
        Args:
            x: State vector
            P: Covariance matrix
            
        Returns:
            sigma_points: 2n+1 x n array of sigma points
        """
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        
        try:
            sqrt = cholesky((self.n + self.lambda_) * P, lower=True)
        except np.linalg.LinAlgError:
            # Handle numerical issues
            sqrt = cholesky((self.n + self.lambda_) * P + np.eye(self.n) * 1e-6, lower=True)
        
        sigma_points[0] = x
        
        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt[:, i]
            sigma_points[i + 1 + self.n] = x - sqrt[:, i]
        
        return sigma_points
    
    def _motion_model(self, state, odometry):
        """
        Apply motion model to a state.
        
        Args:
            state: Current state [x, y, theta]
            odometry: Odometry input [dx, dy, dtheta]
            
        Returns:
            new_state: Predicted state
        """
        dx, dy, dtheta = odometry
        
        cos_theta = np.cos(state[2])
        sin_theta = np.sin(state[2])
        
        new_state = np.array([
            state[0] + dx * cos_theta - dy * sin_theta,
            state[1] + dx * sin_theta + dy * cos_theta,
            state[2] + dtheta
        ])
        
        # Wrap angle to [-pi, pi]
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        
        return new_state
    
    def _measurement_model(self, state, measurement_indices=None):
        """
        Predict laser measurements from a given state.
        
        Args:
            state: State [x, y, theta]
            measurement_indices: Which laser beams to simulate (for efficiency)
            
        Returns:
            predicted_measurements: Array of predicted range measurements
        """
        x, y, theta = state
        
        if measurement_indices is None:
            # Use subset of laser beams for efficiency
            measurement_indices = range(0, 361, 10)  # Every 10th beam
        
        predictions = []
        
        for i in measurement_indices:
            angle = self.laser_angles[i]
            global_angle = theta + angle
            
            # Ray casting to find expected range
            expected_range = self._ray_cast(x, y, global_angle)
            predictions.append(expected_range)
        
        return np.array(predictions)
    
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
        Prediction step using unscented transform.
        
        Args:
            odometry: [dx, dy, dtheta] odometry increments
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points through motion model
        sigma_points_pred = np.zeros_like(sigma_points)
        
        for i in range(2 * self.n + 1):
            sigma_points_pred[i] = self._motion_model(sigma_points[i], odometry)
        
        # Compute predicted mean and covariance
        self.x = np.zeros(self.n)
        for i in range(2 * self.n + 1):
            self.x += self.Wm[i] * sigma_points_pred[i]
        
        # Wrap angle
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        self.P = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            y = sigma_points_pred[i] - self.x
            y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))  # Wrap angle difference
            self.P += self.Wc[i] * np.outer(y, y)
        
        self.P += self.Q
    
    def update(self, laser_measurements):
        """
        Update step using laser range measurements and unscented transform.
        
        Args:
            laser_measurements: Array of 361 laser range measurements
        """
        # Select valid measurements (subset for efficiency)
        measurement_indices = []
        valid_measurements = []
        
        step = 10  # Use every 10th measurement for efficiency
        for i in range(0, len(laser_measurements), step):
            if laser_measurements[i] < self.max_range:
                measurement_indices.append(i)
                valid_measurements.append(laser_measurements[i])
        
        if len(valid_measurements) == 0:
            return  # No valid measurements
        
        z = np.array(valid_measurements)
        m = len(z)  # Number of measurements
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points through measurement model
        Z_pred = np.zeros((2 * self.n + 1, m))
        
        for i in range(2 * self.n + 1):
            Z_pred[i] = self._measurement_model(sigma_points[i], measurement_indices)
        
        # Compute predicted measurement mean and covariance
        z_pred = np.zeros(m)
        for i in range(2 * self.n + 1):
            z_pred += self.Wm[i] * Z_pred[i]
        
        # Innovation covariance
        S = np.zeros((m, m))
        for i in range(2 * self.n + 1):
            y = Z_pred[i] - z_pred
            S += self.Wc[i] * np.outer(y, y)
        
        S += np.eye(m) * self.R
        
        # Cross-correlation matrix
        T = np.zeros((self.n, m))
        for i in range(2 * self.n + 1):
            x_diff = sigma_points[i] - self.x
            z_diff = Z_pred[i] - z_pred
            
            # Handle angle wrapping
            x_diff[2] = np.arctan2(np.sin(x_diff[2]), np.cos(x_diff[2]))
            
            T += self.Wc[i] * np.outer(x_diff, z_diff)
        
        # Kalman gain
        K = T @ np.linalg.inv(S)
        
        # Innovation
        y = z - z_pred
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        self.P = self.P - K @ S @ K.T
        
        # Wrap angle
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()
    
    def get_covariance(self):
        """Return current covariance estimate."""
        return self.P.copy()

def run_ukf_estimation(initial_pose, odometry_data, laser_data):
    """
    Run UKF estimation on the provided data.
    
    Args:
        initial_pose: Initial robot pose (x, y, theta)
        odometry_data: Nx3 array of odometry increments
        laser_data: Nx361 array of laser measurements
    
    Returns:
        Nx3 array of estimated poses
    """
    ukf = UnscentedKalmanFilter(initial_pose)
    
    n_steps = len(odometry_data)
    estimated_poses = np.zeros((n_steps, 3))
    
    # Initialize with first pose
    estimated_poses[0] = initial_pose
    
    # Process each time step
    for i in range(1, n_steps):
        # Predict step
        ukf.predict(odometry_data[i] - odometry_data[i-1])
        
        # Update step with laser measurements
        ukf.update(laser_data[i])
        
        # Store estimate
        estimated_poses[i] = ukf.get_state()
        
        if i % 500 == 0:
            print(f"UKF: Processed {i}/{n_steps} measurements")
    
    return estimated_poses

if __name__ == "__main__":
    # Load data
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    odo_data = np.genfromtxt('odo.csv', delimiter=',')
    laser_data = np.genfromtxt('laser.csv', delimiter=',')
    
    # Initial pose from reference
    initial_pose = (ref_data[0, 0], ref_data[0, 1], ref_data[0, 2])
    
    # Run UKF estimation
    print("Running Unscented Kalman Filter...")
    ukf_poses = run_ukf_estimation(initial_pose, odo_data, laser_data)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    plt.plot(ref_data[:, 0], ref_data[:, 1], 'g-', label='Reference', linewidth=2)
    plt.plot(ukf_poses[:, 0], ukf_poses[:, 1], 'r-', label='UKF Estimate', linewidth=2)
    
    # Mark start and end
    plt.plot(ref_data[0, 0], ref_data[0, 1], 'go', markersize=10, label='Start')
    plt.plot(ref_data[-1, 0], ref_data[-1, 1], 'ro', markersize=10, label='Reference End')
    plt.plot(ukf_poses[-1, 0], ukf_poses[-1, 1], 'bo', markersize=10, label='UKF End')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Unscented Kalman Filter Localization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Calculate and display final error
    final_error = np.sqrt((ref_data[-1, 0] - ukf_poses[-1, 0])**2 + 
                         (ref_data[-1, 1] - ukf_poses[-1, 1])**2)
    plt.text(0.02, 0.98, f'Final Position Error: {final_error:.2f} m', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('ukf_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"UKF Final position error: {final_error:.3f} m")
