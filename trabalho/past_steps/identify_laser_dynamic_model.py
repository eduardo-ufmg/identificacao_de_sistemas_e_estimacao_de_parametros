import argparse
import json

import numpy as np
from sklearn.linear_model import LinearRegression


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


def main():
    parser = argparse.ArgumentParser(
        description="Identify a linear dynamic model from reference and laser data"
    )
    parser.add_argument(
        "--ref", default="ref.csv", help="Path to reference trajectory CSV (x,y,theta)"
    )
    parser.add_argument(
        "--laser", default="laser.csv", help="Path to laser CSV (time x beams)"
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Time step delta for the model (default: 1)"
    )
    parser.add_argument(
        "--save-model", default=None, help="Path to save model parameters (.npz)"
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("Step must be positive")

    ref, laser = load_series(args.ref, args.laser)
    if len(ref) <= args.step:
        raise ValueError("Step is too large for the available data")

    A, B, bias, rmse, model = fit_state_space(ref, laser, args.step)

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


if __name__ == "__main__":
    main()
