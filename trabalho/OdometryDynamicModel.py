import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class OdometryDynamicModel:
    """
    Simulates robot position from odometry deltas in the global frame.
    Assumes odo_diff data contains [dx, dy, dtheta] increments already in global coords.
    """

    def __init__(self, initial_pose: np.ndarray):
        """
        Initialize the model with initial state.

        Args:
            initial_pose: array [x, y, theta]
        """
        self.state = initial_pose.copy()
        self.is_narx = False

    def step(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply one step of odometry delta to the state.

        Args:
            delta: array [dx, dy, dtheta] in global frame

        Returns:
            Updated state [x, y, theta]
        """
        self.state[0] += delta[0]
        self.state[1] += delta[1]
        self.state[2] += delta[2]
        return self.state.copy()

    def simulate(
        self, odo_diff: np.ndarray, true_states: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Simulate the full trajectory from odometry deltas.

        Args:
            odo_diff: shape (n_steps, 3) with [dx, dy, dtheta] per step
            true_states: shape (n_steps + 1, 3) - if provided, performs one-step-ahead
                        prediction by resetting to true state at each step.
                        If None, performs free-run simulation.

        Returns:
            Trajectory array of shape (n_steps + 1, 3) including initial pose
        """
        trajectory = [self.state.copy()]
        for i, delta in enumerate(odo_diff):
            if true_states is not None:
                # One-step-ahead: reset to true state before prediction
                self.state = true_states[i].copy()
            trajectory.append(self.step(delta))
        return np.array(trajectory)


class OdometryNARXModel:
    """
    Simulates robot position from odometry deltas using a NARX (Nonlinear AutoRegressive with eXogenous) model.
    Uses past states and current odometry delta to predict next state.
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

    def build_narx_input(self, odo_delta: np.ndarray) -> np.ndarray:
        """
        Build NARX input from state history and current odometry delta.

        Args:
            odo_delta: Current odometry delta [dx, dy, dtheta]

        Returns:
            Input vector for NARX model
        """
        # Past states (flattened)
        past_states = np.array(self.state_history).flatten()

        # Current odometry delta
        # Concatenate
        x_input = np.concatenate([past_states, odo_delta])

        return x_input

    def step(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply one step of the NARX model.

        Args:
            delta: array [dx, dy, dtheta] in global frame

        Returns:
            Updated state [x, y, theta]
        """
        # Build input
        x_input = self.build_narx_input(delta).reshape(1, -1)

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
        self, odo_diff: np.ndarray, true_states: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Simulate the full trajectory from odometry deltas.

        Args:
            odo_diff: shape (n_steps, 3) with [dx, dy, dtheta] per step
            true_states: shape (n_steps + 1, 3) - if provided, performs one-step-ahead
                        prediction by resetting to true state history at each step.
                        If None, performs free-run simulation.

        Returns:
            Trajectory array of shape (n_steps + 1, 3) including initial pose
        """
        # Start from the most recent state in history
        trajectory = [self.state_history[-1].copy()]

        for i, delta in enumerate(odo_diff):
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

            trajectory.append(self.step(delta))

        return np.array(trajectory)


def load_model(model_json_path: str) -> tuple:
    """Load model parameters from JSON file. Supports simple additive and NARX models."""
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
        # Simple additive model (not used, but kept for compatibility)
        raise ValueError(
            "Only NARX models are supported. Please use --narx when identifying the model."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Simulate robot position from odometry deltas and plot on map"
    )
    parser.add_argument(
        "--odo-diff",
        default="odo_dec_diff_trimmed.csv",
        help="Path to odometry differences CSV (dx, dy, dtheta)",
    )
    parser.add_argument(
        "--map-info", default="map_info.json", help="Path to map info JSON"
    )
    parser.add_argument(
        "--model", default=None, help="Path to NARX model JSON (optional)"
    )
    parser.add_argument(
        "--mode",
        choices=["one-step-ahead", "free-run"],
        default="free-run",
        help="Simulation mode",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Path to reference trajectory CSV (for one-step-ahead mode)",
    )
    args = parser.parse_args()

    # Load map info and image
    with open(args.map_info, "r") as f:
        map_info = json.load(f)

    map_img = Image.open(map_info["image"])
    initial_pose = np.array(map_info["initial_pose"])

    # Load odometry deltas
    odo_diff = np.loadtxt(args.odo_diff, delimiter=",")
    if odo_diff.ndim == 1:
        odo_diff = odo_diff.reshape(-1, 3)

    # Load reference if needed
    true_states = None
    if args.mode == "one-step-ahead":
        if args.reference is None:
            print("Error: --reference is required for one-step-ahead mode")
            return
        ref_data = np.loadtxt(args.reference, delimiter=",")
        if ref_data.ndim == 1:
            ref_data = ref_data.reshape(-1, 3)
        true_states = ref_data[:, :3]

    # Create model
    if args.model:
        # Load NARX model
        is_narx, narx_model, poly_features, narx_config = load_model(args.model)
        model = OdometryNARXModel(narx_model, poly_features, narx_config, initial_pose)
        model_info = (
            f"NARX ({narx_config['model_type']}, n_lags={narx_config['n_lags']})"
        )
        print(f"Using odometry model: {model_info}")
    else:
        # Simple additive model
        model = OdometryDynamicModel(initial_pose)
        model_info = "Simple additive"
        print(f"Using odometry model: {model_info}")

    # Simulate trajectory
    trajectory = model.simulate(odo_diff, true_states)

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

    ax.plot(x_traj, y_traj, ".-", label=f"Odometry Trajectory ({args.mode})")
    ax.plot(x_traj[0], y_traj[0], "o", label="Start")
    ax.plot(x_traj[-1], y_traj[-1], "x", label="End")

    # Plot reference if available
    if true_states is not None:
        ax.plot(
            true_states[:, 0], true_states[:, 1], "k-", alpha=0.3, label="Reference"
        )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    title = f"Robot Trajectory from Odometry - {model_info}"
    ax.set_title(title)
    ax.set_xlim(map_info["xlimits"])
    ax.set_ylim(map_info["ylimits"])
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
