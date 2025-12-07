import numpy as np

class OdometryStaticModel:
    def __init__(self, initial_pose=(0, 0, 0)):
        """
        Initialize the odometry static model.
        
        Args:
            initial_pose: tuple (x, y, theta) representing initial position and orientation
        """
        self.initial_x = initial_pose[0]
        self.initial_y = initial_pose[1]
        self.initial_theta = initial_pose[2]
    
    def compute_positions(self, odometry_data):
        """
        Compute absolute positions from cumulative odometry measurements.
        
        Args:
            odometry_data: Nx3 numpy array with columns [dx_cumulative, dy_cumulative, dtheta_cumulative]
                          where each row contains cumulative distances and angle from the beginning
        
        Returns:
            Nx3 numpy array with columns [x, y, theta] representing absolute positions
        """
        n_points = len(odometry_data)
        positions = np.zeros((n_points, 3))
        
        # First position is the initial pose
        positions[0] = [self.initial_x, self.initial_y, self.initial_theta]
        
        # Compute positions for subsequent points
        for i in range(1, n_points):
            positions[i] = [
                self.initial_x + odometry_data[i, 0],
                self.initial_y + odometry_data[i, 1],
                self.initial_theta + odometry_data[i, 2]
            ]
        
        return positions
    
    @staticmethod
    def load_from_csv(filename):
        """
        Load odometry data from CSV file.
        
        Args:
            filename: path to CSV file with cumulative odometry data
        
        Returns:
            Nx3 numpy array with odometry measurements
        """
        return np.genfromtxt(filename, delimiter=',')
