import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from LaserDynamicModel import LaserDynamicModel, LaserNARXModel, load_model
from OdometryDynamicModel import OdometryDynamicModel


class ParticleFilter:
    """Particle Filter estimator that fuses odometry deltas (control) and pose measurements."""

    def __init__(
        self,
        initial_state: np.ndarray,
        n_particles: int = 1000,
        q_std=(0.02, 0.02, 0.05),
        r_std=(0.05, 0.05, 0.1),
        resample_threshold: float = 0.5,
    ):
        self.initial_state = initial_state.copy()
        self.n_particles = n_particles
        self.q_std = np.array(q_std, dtype=float)
        self.r_std = np.array(r_std, dtype=float)
        self.resample_threshold = resample_threshold

        # Initialize particles around initial state
        self.particles = np.tile(self.initial_state, (self.n_particles, 1))
        self.particles += np.random.randn(self.n_particles, 3) * np.array([0.1, 0.1, 0.1])
        
        # Initialize uniform weights
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, control_delta: np.ndarray) -> None:
        """
        Particle filter prediction step.
        
        Apply control with added process noise to each particle.
        """
        # Add process noise to control
        noise = np.random.randn(self.n_particles, 3) * self.q_std
        self.particles += control_delta + noise

    def update(self, measurement: np.ndarray) -> None:
        """
        Particle filter update step.
        
        Compute weights based on measurement likelihood (Gaussian).
        """
        # Compute innovation for each particle
        innovations = measurement - self.particles
        
        # Compute likelihood (Gaussian)
        # p(z|x) ∝ exp(-0.5 * (z-h(x))^T R^{-1} (z-h(x)))
        R_inv = np.diag(1.0 / np.square(self.r_std))
        
        # Mahalanobis distance for each particle
        mahal_dist = np.sum(innovations @ R_inv * innovations, axis=1)
        
        # Likelihood (unnormalized)
        likelihoods = np.exp(-0.5 * mahal_dist)
        
        # Update weights
        self.weights *= likelihoods
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)  # Normalize

    def resample(self) -> None:
        """
        Resample particles when effective sample size is low.
        Uses systematic resampling for efficiency.
        """
        # Compute effective sample size
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        if n_eff < self.resample_threshold * self.n_particles:
            # Systematic resampling
            positions = (np.arange(self.n_particles) + np.random.rand()) / self.n_particles
            cumulative_sum = np.cumsum(self.weights)
            
            indices = np.searchsorted(cumulative_sum, positions)
            
            # Resample particles
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self) -> np.ndarray:
        """Return weighted mean of particles as state estimate."""
        return np.sum(self.particles * self.weights[:, np.newaxis], axis=0)

    def step(self, control_delta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        """Run one PF predict/update/resample cycle and return the posterior state."""
        self.predict(control_delta)
        self.update(measurement)
        self.resample()
        return self.estimate()

    def run(
        self, 
        odo_deltas: np.ndarray,
        laser_data: np.ndarray,
        laser_model_params: tuple,
    ) -> np.ndarray:
        """Run Particle Filter with one-step-ahead predictions from both odometry and laser models.
        
        Args:
            odo_deltas: odometry deltas (n_steps, 3) - [dx, dy, dtheta] in global frame
            laser_data: laser scans (n_steps, n_beams)
            laser_model_params: tuple from load_model - either (False, A, B, bias) for linear model
                               or (True, narx_model, poly_features, narx_config) for NARX
        """
        is_narx = laser_model_params[0]
        
        if is_narx:
            _, narx_model, poly_features, narx_config = laser_model_params
            laser_model = LaserNARXModel(narx_model, poly_features, narx_config, self.initial_state)
        else:
            _, A, B, bias = laser_model_params
            laser_model = LaserDynamicModel(A, B, bias, self.initial_state)
        
        # Create odometry model for one-step-ahead predictions
        odo_model = OdometryDynamicModel(self.initial_state)
        
        states = [self.estimate()]
        for i, (delta, scan) in enumerate(zip(odo_deltas, laser_data)):
            # One-step-ahead: reset both models to current fused estimate
            current_state = states[i].copy()
            
            # Get odometry prediction from current fused state
            odo_model.state = current_state.copy()
            odo_prediction = odo_model.step(delta)
            control = odo_prediction - current_state  # Delta from current state
            
            # Get laser prediction from current fused state
            laser_model.state = current_state.copy()
            laser_measurement = laser_model.step(scan)
            
            # Fuse predictions
            states.append(self.step(control, laser_measurement))
        return np.array(states)





def load_laser_data(laser_path: str):
    """Load laser scan data from CSV file."""
    laser_data = np.loadtxt(laser_path, delimiter=",")
    if laser_data.ndim == 1:
        laser_data = laser_data.reshape(1, -1)
    return laser_data


def build_odometry_trajectory(odo_diff_path: str, initial_pose: np.ndarray):
    odo_diff = np.loadtxt(odo_diff_path, delimiter=",")
    if odo_diff.ndim == 1:
        odo_diff = odo_diff.reshape(-1, 3)
    model = OdometryDynamicModel(initial_pose)
    return model.simulate(odo_diff)


def load_tuned_params(tuned_path: str):
    """Load tuned parameters from JSON if available; return (n_particles, q_std, r_std, resample_threshold) or None."""
    try:
        with open(tuned_path, "r") as f:
            data = json.load(f)
        n_particles = data.get("n_particles", 1000)
        q_std = tuple(data["q_std"])
        r_std = tuple(data["r_std"])
        resample_threshold = data.get("resample_threshold", 0.5)
        return n_particles, q_std, r_std, resample_threshold
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Particle Filter fusion of odometry deltas and laser-based dynamic model"
    )
    parser.add_argument(
        "--odo-diff",
        default="odo_dec_diff_trimmed.csv",
        help="Path to odometry differences CSV (dx, dy, dtheta)",
    )
    parser.add_argument(
        "--laser", default="laser_dec_trimmed.csv", help="Path to laser data CSV"
    )
    parser.add_argument(
        "--model", default="laser_model.json", help="Path to laser model JSON"
    )
    parser.add_argument(
        "--map-info", default="map_info.json", help="Path to map info JSON"
    )
    parser.add_argument(
        "--tuned",
        default="pf_tuned.json",
        help="Path to tuned parameters JSON (optional)",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Number of particles (overrides tuned if set)",
    )
    parser.add_argument(
        "--q-std",
        type=float,
        nargs=3,
        default=None,
        help="Process noise std for [x, y, theta] (overrides tuned if set)",
    )
    parser.add_argument(
        "--r-std",
        type=float,
        nargs=3,
        default=None,
        help="Measurement noise std for [x, y, theta] (overrides tuned if set)",
    )
    parser.add_argument(
        "--resample-threshold",
        type=float,
        default=None,
        help="Resample when N_eff < threshold * N_particles (overrides tuned if set)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine parameters: prefer CLI args, then tuned, then defaults
    n_particles = args.n_particles
    q_std = args.q_std
    r_std = args.r_std
    resample_threshold = args.resample_threshold

    tuned = load_tuned_params(args.tuned)
    if tuned:
        tuned_n, tuned_q, tuned_r, tuned_resample = tuned
        if n_particles is None:
            n_particles = tuned_n
        if q_std is None:
            q_std = tuned_q
        if r_std is None:
            r_std = tuned_r
        if resample_threshold is None:
            resample_threshold = tuned_resample
        print(f"Loaded tuned parameters from {args.tuned}")
    else:
        if n_particles is None:
            n_particles = 1000
        if q_std is None:
            q_std = (0.02, 0.02, 0.05)
        if r_std is None:
            r_std = (0.05, 0.05, 0.1)
        if resample_threshold is None:
            resample_threshold = 0.5
        print("Using default parameters")

    print(f"Particle Filter with {n_particles} particles")

    # Map and initial pose
    with open(args.map_info, "r") as f:
        map_info = json.load(f)
    map_img = Image.open(map_info["image"])
    initial_pose = np.array(map_info["initial_pose"], dtype=float)

    # Data
    odo_deltas = np.loadtxt(args.odo_diff, delimiter=",")
    if odo_deltas.ndim == 1:
        odo_deltas = odo_deltas.reshape(-1, 3)
    laser_data = load_laser_data(args.laser)
    laser_model_params = load_model(args.model)
    
    # Check model type for display
    is_narx = laser_model_params[0]
    if is_narx:
        _, _, _, narx_config = laser_model_params
        model_info = f"NARX ({narx_config['model_type']}, n_lags={narx_config['n_lags']})"
    else:
        model_info = "Linear model"
    
    print(f"Using laser model: {model_info}")

    # PF Estimation with one-step-ahead predictions from both models
    pf = ParticleFilter(
        initial_pose, 
        n_particles=n_particles, 
        q_std=q_std, 
        r_std=r_std,
        resample_threshold=resample_threshold
    )
    est_states = pf.run(
        odo_deltas=odo_deltas,
        laser_data=laser_data,
        laser_model_params=laser_model_params,
    )
    
    # Build odometry-only trajectory for comparison
    odom_traj = build_odometry_trajectory(args.odo_diff, initial_pose)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    extent = (
        map_info["xlimits"][0],
        map_info["xlimits"][1],
        map_info["ylimits"][0],
        map_info["ylimits"][1],
    )
    ax.imshow(map_img, extent=extent)

    ax.plot(odom_traj[:, 0], odom_traj[:, 1], "b-", label="Odometry only")
    ax.plot(
        est_states[:, 0],
        est_states[:, 1],
        color="orange",
        linestyle="-",
        label=f"PF estimate (N={n_particles})",
    )

    ax.plot(est_states[0, 0], est_states[0, 1], "o", label="Start")
    ax.plot(est_states[-1, 0], est_states[-1, 1], "x", label="End")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    title = "Particle Filter Fusion: Odometry + Laser Dynamic Model"
    if is_narx:
        title += f" - {model_info}"
    ax.set_title(title)
    ax.set_xlim(map_info["xlimits"])
    ax.set_ylim(map_info["ylimits"])
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
