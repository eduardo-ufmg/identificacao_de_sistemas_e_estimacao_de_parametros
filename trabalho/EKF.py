import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for robot localization using odometry and laser range measurements.
    
    State vector: [x, y, theta] - robot position and orientation
    Process model: Simple motion model based on odometry
    Measurement model: Laser range measurements to known landmarks
    """
    
    def __init__(self, initial_pose=(0, 0, 0), process_noise=None, measurement_noise=None):
        """
        Initialize the EKF.
        
        Args:
            initial_pose: Initial robot pose (x, y, theta)
            process_noise: Process noise covariance matrix Q (3x3)
            measurement_noise: Measurement noise variance R
        """
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
            
        # Laser scanner configuration
        self.laser_angles = np.linspace(-np.pi/2, np.pi/2, 361)  # -90° to +90°
        self.max_range = 8.183  # Maximum laser range from data
        
        # Simple landmark map (walls/obstacles for range measurements)
        # These would typically be extracted from a map or SLAM
        self.landmarks = self._create_simple_map()
        
    def _create_simple_map(self):
        """
        Create a simple map with walls for laser range calculations.
        In practice, this would come from a pre-built map or SLAM.
        """
        # Simple rectangular room with walls
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
    
    def predict(self, odometry):
        """
        Prediction step using odometry motion model.
        
        Args:
            odometry: [dx, dy, dtheta] odometry increments
        """
        # Motion model: x_k = f(x_k-1, u_k)
        dx, dy, dtheta = odometry
        
        # Simple motion model (could be improved with proper kinematic model)
        cos_theta = np.cos(self.x[2])
        sin_theta = np.sin(self.x[2])
        
        # State prediction
        self.x[0] += dx * cos_theta - dy * sin_theta
        self.x[1] += dx * sin_theta + dy * cos_theta
        self.x[2] += dtheta
        
        # Wrap angle to [-pi, pi]
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        # Jacobian of motion model with respect to state
        F = np.array([
            [1, 0, -dx * sin_theta - dy * cos_theta],
            [0, 1,  dx * cos_theta - dy * sin_theta],
            [0, 0, 1]
        ])
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, laser_measurements):
        """
        Update step using laser range measurements.
        
        Args:
            laser_measurements: Array of 361 laser range measurements
        """
        valid_measurements = []
        H_list = []
        z_pred_list = []
        
        # Process each laser beam
        for i, (angle, measurement) in enumerate(zip(self.laser_angles, laser_measurements)):
            if measurement < self.max_range:  # Valid measurement
                # Global laser beam angle
                global_angle = self.x[2] + angle
                
                # Expected measurement (ray casting to nearest landmark)
                expected_range = self._ray_cast(self.x[0], self.x[1], global_angle)
                
                if expected_range < self.max_range:
                    # Measurement Jacobian (simplified)
                    # H represents how the measurement changes with state
                    dx_landmark = expected_range * np.cos(global_angle)
                    dy_landmark = expected_range * np.sin(global_angle)
                    
                    H = np.array([
                        -np.cos(global_angle),
                        -np.sin(global_angle),
                        expected_range * np.sin(angle)
                    ]).reshape(1, 3)
                    
                    valid_measurements.append(measurement)
                    z_pred_list.append(expected_range)
                    H_list.append(H)
        
        if len(valid_measurements) == 0:
            return  # No valid measurements
            
        # Stack measurements and Jacobians
        z = np.array(valid_measurements)
        z_pred = np.array(z_pred_list)
        H = np.vstack(H_list)
        
        # Measurement residual
        y = z - z_pred
        
        # Measurement noise covariance
        R = np.eye(len(valid_measurements)) * self.R
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
        
        # Wrap angle
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
    
    def _ray_cast(self, x, y, angle):
        """
        Simple ray casting to find expected range measurement.
        In practice, this would use a proper occupancy grid or map.
        """
        # Simple implementation: find closest landmark in the beam direction
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
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()
    
    def get_covariance(self):
        """Return current covariance estimate."""
        return self.P.copy()

def run_ekf_estimation(initial_pose, odometry_data, laser_data):
    """
    Run EKF estimation on the provided data.
    
    Args:
        initial_pose: Initial robot pose (x, y, theta)
        odometry_data: Nx3 array of odometry increments
        laser_data: Nx361 array of laser measurements
    
    Returns:
        Nx3 array of estimated poses
    """
    ekf = ExtendedKalmanFilter(initial_pose)
    
    n_steps = len(odometry_data)
    estimated_poses = np.zeros((n_steps, 3))
    
    # Initialize with first pose
    estimated_poses[0] = initial_pose
    
    # Process each time step
    for i in range(1, n_steps):
        # Predict step
        ekf.predict(odometry_data[i] - odometry_data[i-1])
        
        # Update step with laser measurements
        ekf.update(laser_data[i])
        
        # Store estimate
        estimated_poses[i] = ekf.get_state()
        
        if i % 500 == 0:
            print(f"EKF: Processed {i}/{n_steps} measurements")
    
    return estimated_poses

if __name__ == "__main__":
    # Load data
    ref_data = np.genfromtxt('ref.csv', delimiter=',')
    odo_data = np.genfromtxt('odo.csv', delimiter=',')
    laser_data = np.genfromtxt('laser.csv', delimiter=',')
    
    # Initial pose from reference
    initial_pose = (ref_data[0, 0], ref_data[0, 1], ref_data[0, 2])
    
    # Run EKF estimation
    print("Running Extended Kalman Filter...")
    ekf_poses = run_ekf_estimation(initial_pose, odo_data, laser_data)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    plt.plot(ref_data[:, 0], ref_data[:, 1], 'g-', label='Reference', linewidth=2)
    plt.plot(ekf_poses[:, 0], ekf_poses[:, 1], 'b-', label='EKF Estimate', linewidth=2)
    
    # Mark start and end
    plt.plot(ref_data[0, 0], ref_data[0, 1], 'go', markersize=10, label='Start')
    plt.plot(ref_data[-1, 0], ref_data[-1, 1], 'ro', markersize=10, label='Reference End')
    plt.plot(ekf_poses[-1, 0], ekf_poses[-1, 1], 'bo', markersize=10, label='EKF End')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Extended Kalman Filter Localization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Calculate and display final error
    final_error = np.sqrt((ref_data[-1, 0] - ekf_poses[-1, 0])**2 + 
                         (ref_data[-1, 1] - ekf_poses[-1, 1])**2)
    plt.text(0.02, 0.98, f'Final Position Error: {final_error:.2f} m', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('ekf_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"EKF Final position error: {final_error:.3f} m")
