import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def build_laser_features(scan: np.ndarray) -> np.ndarray:
    """Extract simple summary features from one laser scan."""
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


def load_series(ref_path: str, laser_path: str):
    ref = np.loadtxt(ref_path, delimiter=",")
    laser = np.loadtxt(laser_path, delimiter=",")
    if ref.ndim != 2 or ref.shape[1] < 3:
        raise ValueError("Reference data must have at least 3 columns (x, y, theta)")
    if laser.ndim != 2:
        raise ValueError("Laser data must be 2-D (time x beams)")
    length = min(len(ref), len(laser))
    if length < 2:
        raise ValueError("Not enough samples to fit a dynamic model")
    return ref[:length], laser[:length]


def fit_state_space(ref: np.ndarray, laser: np.ndarray, step: int):
    states = ref[:, :3]
    states_next = states[step:]
    states_now = states[:-step]
    laser_now = laser[:-step]

    features = np.apply_along_axis(build_laser_features, 1, laser_now)
    X = np.hstack((states_now, features))
    y = states_next

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    coef = model.coef_  # shape (3, features)
    intercept = model.intercept_

    n_state = states_now.shape[1]
    A = coef[:, :n_state]
    B = coef[:, n_state:]

    pred = model.predict(X)
    rmse = np.sqrt(np.mean((pred - y) ** 2, axis=0))

    return A, B, intercept, rmse, model


def fit_mixture_of_experts(ref: np.ndarray, laser: np.ndarray, step: int, n_experts: int, 
                           gating_method: str = 'gmm', max_iter: int = 100, tol: float = 1e-4):
    """
    Fit a mixture of linear experts model.
    
    Args:
        ref: Reference trajectory (n_samples, 3) with [x, y, theta]
        laser: Laser scans (n_samples, n_beams)
        step: Time step delta for the model
        n_experts: Number of expert models
        gating_method: 'gmm' for Gaussian Mixture Model or 'kmeans' for K-Means based gating
        max_iter: Maximum iterations for EM-like optimization
        tol: Convergence tolerance
        
    Returns:
        experts: List of dictionaries with 'A', 'B', 'bias' for each expert
        gating_model: Fitted gating model (GMM or KMeans)
        gating_type: Type of gating used
        rmse: Overall RMSE per state component
        responsibilities: Final soft assignment matrix (n_samples, n_experts)
    """
    states = ref[:, :3]
    states_next = states[step:]
    states_now = states[:-step]
    laser_now = laser[:-step]
    
    features = np.apply_along_axis(build_laser_features, 1, laser_now)
    X = np.hstack((states_now, features))
    y = states_next
    n_samples = X.shape[0]
    n_state = states_now.shape[1]
    
    # Initialize gating network
    if gating_method == 'gmm':
        gating_model = GaussianMixture(n_components=n_experts, covariance_type='full', 
                                       max_iter=100, random_state=0)
        gating_model.fit(X)
        responsibilities = gating_model.predict_proba(X)
    else:  # kmeans
        gating_model = KMeans(n_clusters=n_experts, random_state=0, n_init=10)
        cluster_labels = gating_model.fit_predict(X)
        # Convert hard assignments to soft (one-hot)
        responsibilities = np.zeros((n_samples, n_experts))
        responsibilities[np.arange(n_samples), cluster_labels] = 1.0
    
    # EM-like iterations to refine experts and gating jointly
    experts = []
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        # M-step: Fit weighted linear regression for each expert
        new_experts = []
        for k in range(n_experts):
            weights = responsibilities[:, k]
            
            # Weighted least squares
            W = np.diag(weights)
            X_weighted = np.sqrt(W) @ X
            y_weighted = np.sqrt(W) @ y
            
            # Fit weighted linear regression
            model_k = LinearRegression(fit_intercept=True)
            # Add small ridge for numerical stability
            if np.sum(weights) > 1e-6:
                model_k.fit(X_weighted, y_weighted)
            else:
                # If no samples assigned, use unweighted fit
                model_k.fit(X, y)
            
            coef_k = model_k.coef_
            A_k = coef_k[:, :n_state]
            B_k = coef_k[:, n_state:]
            bias_k = model_k.intercept_
            
            new_experts.append({
                'A': A_k,
                'B': B_k,
                'bias': bias_k,
                'model': model_k
            })
        
        experts = new_experts
        
        # E-step: Update responsibilities based on prediction errors
        predictions = np.zeros((n_samples, n_experts, 3))
        errors = np.zeros((n_samples, n_experts))
        
        for k in range(n_experts):
            pred_k = experts[k]['model'].predict(X)
            predictions[:, k, :] = pred_k
            errors[:, k] = np.sum((y - pred_k) ** 2, axis=1)
        
        # Update responsibilities using softmax on negative squared errors
        # Add small constant for numerical stability
        log_responsibilities = -errors / (2 * np.mean(errors) + 1e-6)
        
        # Add gating network prior
        if gating_method == 'gmm':
            assert isinstance(gating_model, GaussianMixture)
            log_responsibilities += np.log(gating_model.predict_proba(X) + 1e-10)
        
        # Normalize (softmax)
        max_log_resp = np.max(log_responsibilities, axis=1, keepdims=True)
        exp_log_resp = np.exp(log_responsibilities - max_log_resp)
        responsibilities = exp_log_resp / (np.sum(exp_log_resp, axis=1, keepdims=True) + 1e-10)
        
        # Compute loss (weighted sum of squared errors)
        loss = np.sum(responsibilities * errors)
        
        # Check convergence
        if abs(prev_loss - loss) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break
        
        prev_loss = loss
    
    # Compute final predictions and RMSE
    final_predictions = np.sum(responsibilities[:, :, np.newaxis] * predictions, axis=1)
    rmse = np.sqrt(np.mean((y - final_predictions) ** 2, axis=0))
    
    # Print expert statistics
    print(f"\nExpert assignments:")
    for k in range(n_experts):
        n_assigned = np.sum(responsibilities[:, k] > 0.5)
        avg_weight = np.mean(responsibilities[:, k])
        print(f"  Expert {k}: {n_assigned} samples (hard), avg weight: {avg_weight:.3f}")
    
    return experts, gating_model, gating_method, rmse, responsibilities


def main():
    parser = argparse.ArgumentParser(
        description="Identify a linear dynamic model from reference and laser data"
    )
    parser.add_argument(
        "--ref", default="ref_dec.csv", help="Path to reference trajectory CSV (x,y,theta)"
    )
    parser.add_argument(
        "--laser", default="laser_dec.csv", help="Path to laser CSV (time x beams)"
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Time step delta for the model (default: 1)"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Start index for identification range (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index for identification range (default: length)"
    )
    parser.add_argument(
        "--save-model", default=None, help="Path to save model parameters (.npz)"
    )
    parser.add_argument(
        "--map-info", default="map_info.json", help="Path to map info JSON (for visualization)"
    )
    parser.add_argument(
        "--show-plot", action="store_true", help="Show trajectory plot with highlighted range"
    )
    parser.add_argument(
        "--n-experts", type=int, default=1, help="Number of expert models (default: 1 for single model)"
    )
    parser.add_argument(
        "--gating", choices=['gmm', 'kmeans'], default='gmm', 
        help="Gating method: gmm (Gaussian Mixture) or kmeans (K-Means)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Maximum iterations for mixture training"
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("Step must be positive")

    ref, laser = load_series(args.ref, args.laser)
    
    # Parse range
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(ref)
    
    if start_idx < 0 or end_idx > len(ref) or start_idx >= end_idx:
        raise ValueError(f"Invalid range: start={start_idx}, end={end_idx}, length={len(ref)}")
    
    ref_range = ref[start_idx:end_idx]
    laser_range = laser[start_idx:end_idx]
    
    if len(ref_range) <= args.step:
        raise ValueError("Step is too large for the selected range")

    # Fit model(s)
    if args.n_experts > 1:
        # Mixture of experts
        print(f"Fitting mixture of {args.n_experts} linear experts with {args.gating} gating...")
        experts, gating_model, gating_type, rmse, responsibilities = fit_mixture_of_experts(
            ref_range, laser_range, args.step, args.n_experts, 
            gating_method=args.gating, max_iter=args.max_iter
        )
        
        print(f"\nFitted mixture of {args.n_experts} experts on range [{start_idx}, {end_idx}) with step={args.step}")
        print("Model: x_{k+1} = sum_k [ gate_k(x) * (A_k x_k + B_k f(laser_k) + bias_k) ]")
        print(f"\nOverall RMSE per state component [x, y, theta]:")
        print(rmse)
        
        for k, expert in enumerate(experts):
            print(f"\n--- Expert {k} ---")
            print("A matrix (3x3):")
            print(expert['A'])
            print("\nB matrix (3 x features): features = [mean, min, std, left_mean, front_mean, right_mean]")
            print(expert['B'])
            print("\nBias:")
            print(expert['bias'])
        
        # Store mixture info for saving
        A, B, bias, model = None, None, None, None
    else:
        # Single model
        A, B, bias, rmse, model = fit_state_space(ref_range, laser_range, args.step)
        experts, gating_model, gating_type, responsibilities = None, None, None, None
        
        print(f"Fitted linear model on range [{start_idx}, {end_idx}) with step={args.step}")
        print("Fitted linear model: x_{k+1} = A x_k + B f(laser_k) + bias")
        print("A matrix (3x3):")
        print(A)
        print(
            "\nB matrix (3 x features): features = [mean, min, std, left_mean, front_mean, right_mean]"
        )
        print(B)
        print("\nBias:")
        print(bias)
        print("\nRMSE per state component [x, y, theta]:")
        print(rmse)

    if args.save_model:
        if args.n_experts > 1:
            # Save mixture of experts
            save_dict = {
                'n_experts': args.n_experts,
                'gating_type': gating_type,
                'rmse': rmse.tolist(),
                'experts': []
            }
            
            # Save expert parameters
            assert isinstance(experts, list)
            for k, expert in enumerate(experts):
                save_dict['experts'].append({
                    'A': expert['A'].tolist(),
                    'B': expert['B'].tolist(),
                    'bias': expert['bias'].tolist() if isinstance(expert['bias'], np.ndarray) else expert['bias'],
                })
            
            # Save gating model parameters
            if gating_type == 'gmm':
                assert isinstance(gating_model, GaussianMixture)
                save_dict['gating'] = {
                    'means': np.array(gating_model.means_).tolist(),
                    'covariances': np.array(gating_model.covariances_).tolist(),
                    'weights': np.array(gating_model.weights_).tolist(),
                }
            else:  # kmeans
                assert isinstance(gating_model, KMeans)
                save_dict['gating'] = {
                    'centers': gating_model.cluster_centers_.tolist(),
                }
            
            # Save as JSON
            with open(args.save_model + ".json", "w") as f:
                json.dump(save_dict, f, indent=4)
            
            # Save as NPZ with additional info
            npz_dict = {
                'n_experts': args.n_experts,
                'gating_type': gating_type,
                'rmse': rmse,
                'responsibilities': responsibilities,
            }
            for k, expert in enumerate(experts):
                npz_dict[f'A_{k}'] = expert['A']
                npz_dict[f'B_{k}'] = expert['B']
                npz_dict[f'bias_{k}'] = expert['bias']
            
            if gating_type == 'gmm':
                assert isinstance(gating_model, GaussianMixture)
                npz_dict['gating_means'] = gating_model.means_
                npz_dict['gating_covariances'] = gating_model.covariances_
                npz_dict['gating_weights'] = gating_model.weights_
            else:
                assert isinstance(gating_model, KMeans)
                npz_dict['gating_centers'] = gating_model.cluster_centers_
            
            np.savez(args.save_model + ".npz", **npz_dict)
            print(f"Mixture of experts model saved to {args.save_model}.npz and {args.save_model}.json")
        else:
            # Save single model
            assert A is not None and B is not None and bias is not None and model is not None
            np.savez(
                args.save_model + ".npz",
                A=A,
                B=B,
                bias=bias,
                rmse=rmse,
                coef=model.coef_,
                intercept=model.intercept_,
            )

            model_dict = {
                "A": A.tolist(),
                "B": B.tolist(),
                "bias": bias.tolist() if type(bias) is np.ndarray else bias,
                "rmse": rmse.tolist(),
                "coef": model.coef_.tolist(),
                "intercept": (
                    model.intercept_.tolist()
                    if type(model.intercept_) is np.ndarray
                    else model.intercept_
                ),
            }

            with open(args.save_model + ".json", "w") as f:
                json.dump(model_dict, f, indent=4)

            print(f"Model saved to {args.save_model}.npz and {args.save_model}.json")

    if args.show_plot:
        # Load map info
        try:
            with open(args.map_info, 'r') as f:
                map_info = json.load(f)
            map_img = Image.open(map_info['image'])
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            extent = (
                map_info['xlimits'][0],
                map_info['xlimits'][1],
                map_info['ylimits'][0],
                map_info['ylimits'][1],
            )
            ax.imshow(map_img, extent=extent)
            
            # Plot full trajectory in light gray
            ax.plot(ref[:, 0], ref[:, 1], color='lightgray', linewidth=1, label='Full trajectory')
            
            # Highlight selected range in color
            if args.n_experts > 1 and responsibilities is not None:
                # Color by dominant expert
                dominant_expert = np.argmax(responsibilities, axis=1)
                colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, args.n_experts))
                
                for k in range(args.n_experts):
                    mask = dominant_expert == k
                    if np.any(mask):
                        ax.scatter(ref_range[:-args.step][mask, 0], ref_range[:-args.step][mask, 1], 
                                  c=[colors[k]], s=20, alpha=0.6, label=f'Expert {k}')
            else:
                ax.plot(ref_range[:, 0], ref_range[:, 1], 'b-', linewidth=2, label=f'Selected range [{start_idx}, {end_idx})')
            
            # Mark range boundaries
            ax.plot(ref_range[0, 0], ref_range[0, 1], 'go', markersize=10, label='Range start')
            ax.plot(ref_range[-1, 0], ref_range[-1, 1], 'ro', markersize=10, label='Range end')
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            title = f'Reference Trajectory - Identification Range [{start_idx}, {end_idx})'
            if args.n_experts > 1:
                title += f' ({args.n_experts} Experts)'
            ax.set_title(title)
            ax.set_xlim(map_info['xlimits'])
            ax.set_ylim(map_info['ylimits'])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            plt.tight_layout()
            plt.show()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load map for visualization: {e}")


if __name__ == "__main__":
    main()
