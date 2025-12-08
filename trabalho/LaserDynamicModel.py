import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

    def simulate(
        self, laser_data: np.ndarray, true_states: np.ndarray | None = None
    ) -> np.ndarray:
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


class LaserNARXModel:
    """
    Simulates robot position from laser scans using a NARX (Nonlinear AutoRegressive with eXogenous) model.
    Uses past states and current laser features to predict next state.
    """

    def __init__(
        self, narx_model, poly_features, narx_config: dict, initial_pose: np.ndarray
    ):
        """
        Initialize the NARX model.

        Args:
            narx_model: Fitted sklearn model (Ridge, MLPRegressor, etc.)
            poly_features: PolynomialFeatures object (or None)
            narx_config: Configuration dictionary with 'n_lags', 'step', 'model_type', etc.
            initial_pose: initial state [x, y, theta]
        """
        self.narx_model = narx_model
        self.poly_features = poly_features
        self.config = narx_config
        self.n_lags = narx_config["n_lags"]
        self.step_size = narx_config["step"]  # Renamed to avoid shadowing step() method
        self.model_type = narx_config["model_type"]

        # Initialize state history with copies of initial pose
        self.state_history = [initial_pose.copy() for _ in range(self.n_lags)]
        self.is_narx = True

    @property
    def state(self) -> np.ndarray:
        """Return the current state (most recent in history)."""
        return self.state_history[-1].copy()

    @state.setter
    def state(self, new_state: np.ndarray) -> None:
        """Set the current state (most recent in history)."""
        self.state_history[-1] = new_state.copy()

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

    def build_narx_input(self, laser_scan: np.ndarray) -> np.ndarray:
        """
        Build NARX input from state history and current laser scan.

        Args:
            laser_scan: Current laser scan

        Returns:
            Input vector for NARX model
        """
        # Past states (flattened)
        past_states = np.array(self.state_history).flatten()

        # Current laser features
        laser_features = self.extract_laser_features(laser_scan)

        # Concatenate
        x_input = np.concatenate([past_states, laser_features])

        return x_input

    def step(self, laser_scan: np.ndarray) -> np.ndarray:
        """
        Apply one step of the NARX model.

        Args:
            laser_scan: 1-D array of laser ranges

        Returns:
            Updated state [x, y, theta]
        """
        # Build input
        x_input = self.build_narx_input(laser_scan).reshape(1, -1)

        # Apply polynomial features if needed
        if self.poly_features is not None:
            x_input = self.poly_features.transform(x_input)

        # Predict
        next_state = self.narx_model.predict(x_input)[0]

        # Update state history (shift left and append new state)
        self.state_history.pop(0)
        self.state_history.append(next_state.copy())

        return next_state

    def simulate(
        self, laser_data: np.ndarray, true_states: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Simulate the full trajectory from laser scans.

        Args:
            laser_data: shape (n_steps, n_beams)
            true_states: shape (n_steps + 1, 3) - if provided, performs one-step-ahead
                        prediction by resetting to true state history at each step.
                        If None, performs free-run simulation.

        Returns:
            Trajectory array of shape (n_steps + 1, 3) including initial pose
        """
        # Start from the most recent state in history
        trajectory = [self.state_history[-1].copy()]

        for i, scan in enumerate(laser_data):
            if true_states is not None:
                # One-step-ahead: reset state history to true states
                # Use states from i to i+n_lags-1 (or available)
                start_idx = max(0, i - self.n_lags + 1)
                end_idx = i + 1

                # Pad with initial states if needed
                if start_idx == 0 and end_idx < self.n_lags:
                    self.state_history = [
                        true_states[0].copy() for _ in range(self.n_lags - end_idx)
                    ]
                    self.state_history.extend(
                        [true_states[j].copy() for j in range(start_idx, end_idx)]
                    )
                else:
                    self.state_history = [
                        true_states[j].copy() for j in range(start_idx, end_idx)
                    ]
                    # Pad to n_lags if needed
                    while len(self.state_history) < self.n_lags:
                        self.state_history.insert(0, true_states[0].copy())

            trajectory.append(self.step(scan))

        return np.array(trajectory)


def load_model(model_json_path: str) -> tuple:
    """Load model parameters from JSON file. Supports simple linear and NARX models."""
    with open(model_json_path, "r") as f:
        data = json.load(f)

    # Check if it's a NARX model
    if "model_type" in data and data["model_type"] == "narx":
        # Load NARX model from pickle file
        pkl_path = model_json_path.replace(".json", ".pkl")
        try:
            with open(pkl_path, "rb") as f:
                model_data = pickle.load(f)

            narx_model = model_data["narx_model"]
            poly_features = model_data["poly_features"]
            narx_config = model_data["narx_config"]

            return True, narx_model, poly_features, narx_config
        except FileNotFoundError:
            raise FileNotFoundError(f"NARX model requires pickle file: {pkl_path}")
    else:
        # Simple linear model
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
        "--ref",
        default=None,
        help="Path to reference trajectory CSV for one-step-ahead prediction",
    )
    parser.add_argument(
        "--mode",
        choices=["free-run", "one-step"],
        default="free-run",
        help="Simulation mode: free-run or one-step-ahead",
    )
    args = parser.parse_args()

    # Load model parameters
    model_data = load_model(args.model)
    is_narx = model_data[0]

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
    if is_narx:
        _, narx_model, poly_features, narx_config = model_data
        model = LaserNARXModel(narx_model, poly_features, narx_config, initial_pose)
        model_type = f"NARX ({narx_config['model_type']}, lags={narx_config['n_lags']})"
    else:
        _, A, B, bias = model_data
        model = LaserDynamicModel(A, B, bias, initial_pose)
        model_type = "Linear"

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
        ax.plot(
            true_states[:, 0], true_states[:, 1], "g--", alpha=0.5, label="Reference"
        )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    title = f"Robot Trajectory from Laser Dynamic Model ({mode_label})"
    if is_narx:
        title += f" - {model_type}"
    ax.set_title(title)
    ax.set_xlim(map_info["xlimits"])
    ax.set_ylim(map_info["ylimits"])
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
