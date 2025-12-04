import numpy as np

class OdometryDynamicModel:
    def __init__(self, initial_pose=(0.0, 0.0, 0.0)):
        """
        Initialize the dynamic odometry model that integrates differential motions.

        Args:
            initial_pose: tuple (x, y, theta) representing initial position and orientation
                          theta in radians.
        """
        self.initial_x = float(initial_pose[0])
        self.initial_y = float(initial_pose[1])
        self.initial_theta = float(initial_pose[2])

    def compute_positions(self, differential_data: np.ndarray) -> np.ndarray:
        """
        Integrate differential odometry measurements to obtain absolute positions.

        Args:
            differential_data: Nx3 numpy array with columns [dx, dy, dtheta]
                               representing incremental motion between consecutive timestamps.
                               dx, dy are assumed to be measured in the robot local frame
                               (forward-right coordinates), dtheta in radians.

        Returns:
            Nx3 numpy array with columns [x, y, theta] in the global/world frame.
            The first row corresponds to the initial pose; subsequent rows result
            from integrating the increments.
        """
        if differential_data.ndim != 2 or differential_data.shape[1] < 3:
            raise ValueError("differential_data must be an Nx3 array")

        n_points = differential_data.shape[0]
        positions = np.zeros((n_points, 3), dtype=float)

        # Set first pose as the initial pose
        positions[0, 0] = self.initial_x
        positions[0, 1] = self.initial_y
        positions[0, 2] = self.initial_theta

        x = self.initial_x
        y = self.initial_y
        theta = self.initial_theta

        # Integrate each incremental motion
        for i in range(1, n_points):
            dx_local = float(differential_data[i, 0])
            dy_local = float(differential_data[i, 1])
            dtheta = float(differential_data[i, 2])

            # Update position in global frame
            x += dx_local
            y += dy_local
            theta += dtheta

            positions[i, 0] = x
            positions[i, 1] = y
            positions[i, 2] = theta

        return positions

    @staticmethod
    def load_from_csv(filename: str) -> np.ndarray:
        """
        Load differential odometry data from CSV file.

        Args:
            filename: path to CSV file with differential odometry data (e.g., 'odo_diff.csv')

        Returns:
            Nx3 numpy array with incremental motions [dx, dy, dtheta]
        """
        data = np.genfromtxt(filename, delimiter=',')
        # Ensure 2D shape if a single row
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
