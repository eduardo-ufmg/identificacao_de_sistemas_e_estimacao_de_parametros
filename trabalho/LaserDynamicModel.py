import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class LaserDynamicModel:
    """
    Simulates robot position from laser scans using a linear dynamic model.
    Model: x_{k+1} = A x_k + B f(laser_k) + bias
    where x = [px, py, theta], f(laser) = [mean, min, std, left_mean, front_mean, right_mean]
    """

    def __init__(
        self, A: np.ndarray, B: np.ndarray, bias: np.ndarray, initial_pose: np.ndarray
    ):
        """
        Initialize the model.

        Args:
            A: state transition matrix (3x3)
            B: laser feature matrix (3x6)
            bias: bias vector (3,)
            initial_pose: initial state [x, y, theta]
        """
        self.A = A
        self.B = B
        self.bias = bias
        self.state = initial_pose.copy()
        self.is_mixture = False

    @staticmethod
    def extract_laser_features(scan: np.ndarray) -> np.ndarray:
        """Extract features from one laser scan."""
        n = scan.size
        if n == 0:
            return np.zeros(6, dtype=float)
        third = max(1, n // 3)
        left = scan[:third]
        front = scan[third : 2 * third]
        right = scan[2 * third :]
        return np.array(
            [
                scan.mean(),
                scan.min(),
                scan.std(),
                left.mean(),
                front.mean() if front.size else scan.mean(),
                right.mean() if right.size else scan.mean(),
            ],
            dtype=float,
        )

    def step(self, laser_scan: np.ndarray) -> np.ndarray:
        """
        Apply one step of the dynamic model.

        Args:
            laser_scan: 1-D array of laser ranges

        Returns:
            Updated state [x, y, theta]
        """
        features = self.extract_laser_features(laser_scan)
        self.state = self.A @ self.state + self.B @ features + self.bias
        return self.state.copy()

    def simulate(self, laser_data: np.ndarray, true_states: np.ndarray | None = None) -> np.ndarray:
        """
        Simulate the full trajectory from laser scans.

        Args:
            laser_data: shape (n_steps, n_beams)
            true_states: shape (n_steps + 1, 3) - if provided, performs one-step-ahead
                        prediction by resetting to true state at each step.
                        If None, performs free-run simulation.

        Returns:
            Trajectory array of shape (n_steps + 1, 3) including initial pose
        """
        trajectory = [self.state.copy()]
        for i, scan in enumerate(laser_data):
            if true_states is not None:
                # One-step-ahead: reset to true state before prediction
                self.state = true_states[i].copy()
            trajectory.append(self.step(scan))
        return np.array(trajectory)


class LaserMixtureModel:
    """
    Simulates robot position from laser scans using a mixture of linear experts.
    Each expert is a linear dynamic model, and a gating network determines their contributions.
    """

    def __init__(
        self,
        experts: list,
        gating_model,
        gating_type: str,
        initial_pose: np.ndarray
    ):
        """
        Initialize the mixture model.

        Args:
            experts: List of dicts with 'A', 'B', 'bias' for each expert
            gating_model: Fitted gating model (GaussianMixture or KMeans)
            gating_type: 'gmm' or 'kmeans'
            initial_pose: initial state [x, y, theta]
        """
        self.experts = experts
        self.gating_model = gating_model
        self.gating_type = gating_type
        self.n_experts = len(experts)
        self.state = initial_pose.copy()
        self.is_mixture = True

    @staticmethod
    def extract_laser_features(scan: np.ndarray) -> np.ndarray:
        """Extract features from one laser scan."""
        n = scan.size
        if n == 0:
            return np.zeros(6, dtype=float)
        third = max(1, n // 3)
        left = scan[:third]
        front = scan[third : 2 * third]
        right = scan[2 * third :]
        return np.array(
            [
                scan.mean(),
                scan.min(),
                scan.std(),
                left.mean(),
                front.mean() if front.size else scan.mean(),
                right.mean() if right.size else scan.mean(),
            ],
            dtype=float,
        )

    def compute_gating_weights(self, state: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Compute gating weights for each expert based on current state and features.
        
        Args:
            state: current state [x, y, theta]
            features: laser features
            
        Returns:
            weights: array of shape (n_experts,) with weights summing to 1
        """
        # Concatenate state and features as input to gating network
        x_input = np.concatenate([state, features]).reshape(1, -1)
        
        if self.gating_type == 'gmm':
            # Use GMM posterior probabilities
            weights = self.gating_model.predict_proba(x_input)[0]
        else:  # kmeans
            # Use distance-based soft assignment
            distances = np.linalg.norm(
                self.gating_model.cluster_centers_ - x_input, axis=1
            )
            # Convert distances to weights using softmax
            inv_distances = 1.0 / (distances + 1e-6)
            weights = inv_distances / np.sum(inv_distances)
        
        return weights

    def step(self, laser_scan: np.ndarray) -> np.ndarray:
        """
        Apply one step of the mixture model.

        Args:
            laser_scan: 1-D array of laser ranges

        Returns:
            Updated state [x, y, theta]
        """
        features = self.extract_laser_features(laser_scan)
        weights = self.compute_gating_weights(self.state, features)
        
        # Weighted combination of expert predictions
        next_state = np.zeros(3, dtype=float)
        for k, expert in enumerate(self.experts):
            A_k = np.array(expert['A'])
            B_k = np.array(expert['B'])
            bias_k = np.array(expert['bias'])
            
            pred_k = A_k @ self.state + B_k @ features + bias_k
            next_state += weights[k] * pred_k
        
        self.state = next_state
        return self.state.copy()

    def simulate(self, laser_data: np.ndarray, true_states: np.ndarray | None = None) -> np.ndarray:
        """
        Simulate the full trajectory from laser scans.

        Args:
            laser_data: shape (n_steps, n_beams)
            true_states: shape (n_steps + 1, 3) - if provided, performs one-step-ahead
                        prediction by resetting to true state at each step.
                        If None, performs free-run simulation.

        Returns:
            Trajectory array of shape (n_steps + 1, 3) including initial pose
        """
        trajectory = [self.state.copy()]
        for i, scan in enumerate(laser_data):
            if true_states is not None:
                # One-step-ahead: reset to true state before prediction
                self.state = true_states[i].copy()
            trajectory.append(self.step(scan))
        return np.array(trajectory)


def load_model(model_json_path: str) -> tuple:
    """Load model parameters from JSON file. Supports both single and mixture models."""
    with open(model_json_path, "r") as f:
        data = json.load(f)
    
    # Check if it's a mixture model
    if 'n_experts' in data and data['n_experts'] > 1:
        # Load mixture of experts
        experts = data['experts']
        gating_type: str = data['gating_type']
        
        # Reconstruct gating model
        if gating_type == 'gmm':
            n_components = data['n_experts']
            gating_model = GaussianMixture(n_components=n_components, covariance_type='full')
            # Manually set parameters
            gating_model.means_ = np.array(data['gating']['means'])
            gating_model.covariances_ = np.array(data['gating']['covariances'])
            gating_model.weights_ = np.array(data['gating']['weights'])
            gating_model.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(gating_model.covariances_)
            )
        else:  # kmeans
            n_clusters = data['n_experts']
            gating_model = KMeans(n_clusters=n_clusters)
            gating_model.cluster_centers_ = np.array(data['gating']['centers'])
        
        return True, experts, gating_model, gating_type
    else:
        # Single model
        A = np.array(data["A"])
        B = np.array(data["B"])
        bias = np.array(data["bias"])
        return False, A, B, bias


def main():
    parser = argparse.ArgumentParser(
        description="Simulate robot position from laser scans using fitted dynamic model"
    )
    parser.add_argument(
        "--laser", default="laser_dec_trimmed.csv", help="Path to laser data CSV"
    )
    parser.add_argument(
        "--model", default="laser_model.json", help="Path to model parameters JSON"
    )
    parser.add_argument(
        "--map-info", default="map_info.json", help="Path to map info JSON"
    )
    parser.add_argument(
        "--ref", default=None, help="Path to reference trajectory CSV for one-step-ahead prediction"
    )
    parser.add_argument(
        "--mode", choices=["free-run", "one-step"], default="free-run",
        help="Simulation mode: free-run or one-step-ahead"
    )
    args = parser.parse_args()

    # Load model parameters
    model_data = load_model(args.model)
    is_mixture = model_data[0]

    # Load map info
    with open(args.map_info, "r") as f:
        map_info = json.load(f)

    map_img = Image.open(map_info["image"])
    initial_pose = np.array(map_info["initial_pose"])

    # Load laser data
    laser_data = np.loadtxt(args.laser, delimiter=",")
    if laser_data.ndim == 1:
        laser_data = laser_data.reshape(1, -1)

    # Load reference trajectory if one-step-ahead mode
    true_states = None
    if args.mode == "one-step" or args.ref:
        ref_path = args.ref if args.ref else "ref_dec.csv"
        true_states = np.loadtxt(ref_path, delimiter=",")
        if true_states.ndim == 1:
            true_states = true_states.reshape(1, -1)

    # Create and simulate model
    if is_mixture:
        _, experts, gating_model, gating_type = model_data
        model = LaserMixtureModel(list(experts), gating_model, gating_type, initial_pose)
        model_type = f"Mixture ({len(experts)} experts)"
    else:
        _, A, B, bias = model_data
        model = LaserDynamicModel(A, B, bias, initial_pose)
        model_type = "Single"
    
    trajectory = model.simulate(laser_data, true_states)

    x_traj = trajectory[:, 0]
    y_traj = trajectory[:, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    extent = (
        map_info["xlimits"][0],
        map_info["xlimits"][1],
        map_info["ylimits"][0],
        map_info["ylimits"][1],
    )
    ax.imshow(map_img, extent=extent)

    mode_label = "One-Step-Ahead" if true_states is not None else "Free-Run"
    ax.plot(x_traj, y_traj, ".-", label=f"Laser-based Trajectory ({mode_label})")
    ax.plot(x_traj[0], y_traj[0], "o", label="Start")
    ax.plot(x_traj[-1], y_traj[-1], "x", label="End")

    # Plot reference trajectory if available
    if true_states is not None:
        ax.plot(true_states[:, 0], true_states[:, 1], "g--", alpha=0.5, label="Reference")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    title = f"Robot Trajectory from Laser Dynamic Model ({mode_label})"
    if is_mixture:
        title += f" - {model_type}"
    ax.set_title(title)
    ax.set_xlim(map_info["xlimits"])
    ax.set_ylim(map_info["ylimits"])
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
