#!/usr/bin/env python3

"""
Performs state estimation for Chua's circuit using EKF and UKF.

This script loads experimental data from 'pcchua_dados.dat' or 'pcchua_pert.dat',
defines the nonlinear system dynamics, and applies both an Extended Kalman Filter (EKF)
and an Unscented Kalman Filter (UKF) to estimate the unmeasured state v_C1 (x1).

It also provides plots to analyze the estimation accuracy and check the
"whiteness" of the innovations (residuals) via an autocorrelation plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# =============================================================================
# 1. Parameters and System Model
# =============================================================================

# --- System Parameters (in SI units) ---
C1 = 30.14e-6  # F
C2 = 185.6e-6  # F
L = 52.28      # H
R_res = 1673.0 # Ohms (series resistor between C1 and C2)
RL = 0.0       # Ohms
Ga = -0.801e-3 # S (mS to S)
Gb = -0.365e-3 # S (mS to S)
E = 1.74       # V
Ts = 0.01      # s (Sampling time 10.00 ms)

# --- Nonlinear Diode Function i_N(x1) ---
# Piecewise-linear current i_N = G(v_C1) * v_C1
def i_N_func(x1):
    """
    Calculates the nonlinear diode current i_N(x1).
    x1 is v_C1.
    """
    if abs(x1) <= E:
        return Ga * x1
    elif x1 > E:
        return Gb * x1 + (Ga - Gb) * E
    else: # x1 < -E
        return Gb * x1 - (Ga - Gb) * E

# --- Derivative of Nonlinearity g(x1) ---
def g_func(x1):
    """
    Calculates the derivative d(i_N)/d(x1).
    """
    if abs(x1) < E:
        return Ga
    else:
        return Gb

# --- Continuous-Time State Equations ---
def f_continuous(x, u):
    """
    Continuous-time state-space model: dx/dt = f(x, u)
    x = [x1, x2, x3] = [v_C1, v_C2, i_L]
    u = [u1, u2, u3] = [Tx, Ty, rz]
    """
    x1, x2, x3 = x
    u1, u2, u3 = u
    
    i_N = i_N_func(x1) # Nonlinear current
    
    dx1_dt = (1/C1) * ((x2 - x1)/R_res - i_N + u1)
    dx2_dt = (1/C2) * ((x1 - x2)/R_res + x3 + u2)
    dx3_dt = (1/L) * (-x2 + RL * x3 + u3)
    
    return np.array([dx1_dt, dx2_dt, dx3_dt])

# --- Discrete-Time State Transition (Forward Euler) ---
def f_discrete_euler(x, dt, u=None):
    """
    Discrete-time state transition function: x_{k+1} = F(x_k, u_k)
    Uses Forward Euler: x_{k+1} = x_k + dt * f(x_k, u_k)

    Signature compatible with filterpy's UKF: fx(x, dt, **fx_args)
    where we pass control input via keyword arg 'u'.
    """
    if u is None:
        u = np.zeros(3)
    return x + dt * f_continuous(x, u)

# --- Measurement Function ---
def h_measurement(x):
    """
    Measurement function: y_k = h(x_k)
    We measure y = [v_C2, i_L] = [x2, x3]
    """
    return np.array([x[1], x[2]])

# Measurement Jacobian (H) - Constant since h is linear
H_matrix = np.array([
    [0., 1., 0.],
    [0., 0., 1.]
])

# --- EKF State Transition Jacobian (A) ---
def get_jacobian_A(x):
    """
    Calculates the Jacobian of the discrete state transition function F
    w.r.t. state x:  A = dF/dx
    
    A = I + Ts * (df/dx)
    
    df/dx = [ [ 1/C1*(-1/R_res - g(x1)), 1/(C1*R_res)      , 0     ],
              [ 1/(C2*R_res)            , -1/(C2*R_res)     , 1/C2  ],
              [ 0                   , -1/L          , RL/L  ] ]
    """
    x1 = x[0]
    
    # Jacobian of continuous f
    J_f = np.array([
        [(1/C1) * (-1/R_res - g_func(x1)), 1/(C1*R_res) , 0.   ],
        [1/(C2*R_res)                    , -1/(C2*R_res), 1/C2   ],
        [0.                          , -1/L     , RL/L ]
    ])
    
    # Jacobian of discrete F (A = I + Ts * J_f)
    A = np.eye(3) + Ts * J_f
    return A

# =============================================================================
# 2. Data Loading
# =============================================================================

def load_chua_data(filename):
    """
    Loads data from the specified .dat file.
    """
    # Column names based on the file header
    col_names = ['time', 'x', 'y', 'z', 'xref', 'yref', 'zref', 'ux', 'uy', 'uz']
    
    # Load the data using pandas, skipping comments and using whitespace
    data = pd.read_csv(
        filename,
        comment='#',
        sep='\\s+',
        header=None,
        names=col_names
    )
    
    # Extract states x = [v_C1, v_C2, i_L]
    true_states = data[['x', 'y', 'z']].values
    
    # Extract measurements z = [v_C2, i_L]
    measurements = data[['y', 'z']].values
    
    # Extract inputs u = [Tx, Ty, rz]
    inputs = data[['ux', 'uy', 'uz']].values
    
    time = data['time'].values
    
    print(f"Loaded {len(data)} data points from {filename}.")
    return time, true_states, measurements, inputs

# =============================================================================
# 3. Extended Kalman Filter (EKF) Implementation
# =============================================================================

def run_ekf(z_data, u_data, Q, R, x0, P0):
    """
    Runs the Extended Kalman Filter on the provided data.
    
    Args:
        z_data (np.array): Measurement data (N_samples, 2)
        u_data (np.array): Input data (N_samples, 3)
        Q (np.array): Process noise covariance (3x3)
        R (np.array): Measurement noise covariance (2x2)
        x0 (np.array): Initial state estimate (3,)
        P0 (np.array): Initial state covariance (3x3)
        
    Returns:
        Tuple of (state_estimates, covariance_estimates, innovations)
    """
    N_samples = z_data.shape[0]
    
    # Arrays to store results
    x_est = np.zeros((N_samples, 3))
    P_est = np.zeros((N_samples, 3, 3))
    innovations = np.zeros((N_samples, 2))
    
    # Set initial conditions
    x_k = x0
    P_k = P0
    
    for k in range(N_samples):
        # --- PREDICT ---
        # Project state ahead
        x_pred = f_discrete_euler(x_k, Ts, u=u_data[k])
        
        # Get state transition Jacobian
        A_k = get_jacobian_A(x_k)
        
        # Project covariance ahead
        P_pred = A_k @ P_k @ A_k.T + Q
        
        # --- UPDATE ---
        # Get measurement y_k (innovation)
        y_k = z_data[k] - h_measurement(x_pred)
        
        # Get innovation covariance
        S_k = H_matrix @ P_pred @ H_matrix.T + R
        
        # Get Kalman gain
        K_k = P_pred @ H_matrix.T @ np.linalg.inv(S_k)
        
        # Update state estimate
        x_k = x_pred + K_k @ y_k
        
        # Update covariance estimate
        P_k = (np.eye(3) - K_k @ H_matrix) @ P_pred
        
        # Store results
        x_est[k] = x_k
        P_est[k] = P_k
        innovations[k] = y_k
        
    return x_est, P_est, innovations

# =============================================================================
# 4. Unscented Kalman Filter (UKF) Implementation
# =============================================================================

def run_ukf(z_data, u_data, Q, R, x0, P0):
    """
    Runs the Unscented Kalman Filter on the provided data using filterpy.
    """
    N_samples = z_data.shape[0]
    
    # Define sigma points
    # (n=3 states, alpha=0.1 (good default), beta=2. (optimal for Gaussian), kappa=0.)
    points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2., kappa=0.)
    
    # Create UKF
    ukf = UnscentedKalmanFilter(
        dim_x=3,           # 3 states
        dim_z=2,           # 2 measurements
        dt=Ts,
        fx=f_discrete_euler, # Discrete state transition function
        hx=h_measurement,    # Measurement function
        points=points
    )
    
    # Set initial conditions
    ukf.x = x0
    ukf.P = P0
    
    # Set noise covariances
    ukf.Q = Q
    ukf.R = R
    
    # Arrays to store results
    x_est = np.zeros((N_samples, 3))
    P_est = np.zeros((N_samples, 3, 3))
    innovations = np.zeros((N_samples, 2))
    
    for k in range(N_samples):
        # Predict step. Pass input u_data[k] to fx.
        ukf.predict(u=u_data[k])
        
        # Update step
        ukf.update(z=z_data[k])
        
        # Store results
        x_est[k] = ukf.x.copy()
        P_est[k] = ukf.P.copy()
        innovations[k] = ukf.y.copy() # .y is the innovation (residual)
        
    return x_est, P_est, innovations

# =============================================================================
# 5. Helper Functions for Analysis
# =============================================================================

def plot_results(time, true_x1, estimates, P_estimates, title):
    """
    Plots the true vs estimated v_C1 (x1) and the estimation error.
    """
    est_x1 = estimates[:, 0]
    err = true_x1 - est_x1
    
    # Get 3-sigma confidence bounds
    std_x1 = np.sqrt(P_estimates[:, 0, 0])
    upper_bound = est_x1 + 3 * std_x1
    lower_bound = est_x1 - 3 * std_x1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # --- Plot 1: True vs Estimated State ---
    ax1.plot(time, true_x1, 'k-', label='True $v_{C1} (x_1)$')
    ax1.plot(time, est_x1, 'r--', label='Estimated $v_{C1}$')
    ax1.fill_between(time, lower_bound, upper_bound, color='r', alpha=0.2, label='3-sigma Confidence')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend()
    ax1.grid(True)
    
    # --- Plot 2: Estimation Error ---
    ax2.plot(time, err, 'b-', label='Error ($v_{C1} - \\hat{v}_{C1}$)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (V)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

def plot_innovations_and_acf(innovations, dt, title):
    """
    Plots the innovations (residuals) and their autocorrelation.
    """
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=16)
    
    time = np.arange(len(innovations)) * dt
    
    # --- Plot 1: Innovation for y1 (v_C2) ---
    ax1[0].plot(time, innovations[:, 0])
    ax1[0].set_title('Innovation for $y_1$ ($v_{C2}$)')
    ax1[0].set_ylabel('Innovation (V)')
    ax1[0].grid(True)
    
    # --- Plot 2: Innovation for y2 (i_L) ---
    ax1[1].plot(time, innovations[:, 1])
    ax1[1].set_title('Innovation for $y_2$ ($i_L$)')
    ax1[1].set_ylabel('Innovation (A)')
    ax1[1].grid(True)
    
    # --- Plot 3: ACF for y1 ---
    plot_acf(innovations[:, 0], lags=50, ax=ax2[0], title='ACF for $y_1$ Innovation')
    ax2[0].set_xlabel('Lags')
    
    # --- Plot 4: ACF for y2 ---
    plot_acf(innovations[:, 1], lags=50, ax=ax2[1], title='ACF for $y_2$ Innovation')
    ax2[1].set_xlabel('Lags')
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

# =============================================================================
# 6. Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # --- Configuration ---
    FILENAME = 'pcchua_dados.dat'  # or 'pcchua_pert.dat'
    
    # --- Load Data ---
    time, true_states, measurements, inputs = load_chua_data(FILENAME)
    
    # --- Tuning Parameters ---
    
    # Process Noise Covariance (Q)
    # How much do we trust our model? (dx/dt = f(x,u))
    # Higher values = model is noisy/inaccurate.
    # Start with small values.
    q_val = 1e-7
    Q = np.diag([q_val, q_val, q_val])
    
    # Measurement Noise Covariance (R)
    # How much do we trust our measurements? (y = [v_C2, i_L])
    # Higher values = sensors are noisy.
    # We can estimate this from the sensor datasheet or data.
    r_val_v = 1e-4  # Variance for v_C2
    r_val_i = 1e-5  # Variance for i_L
    R = np.diag([r_val_v, r_val_i])
    
    # --- Initial Conditions ---
    
    # Initial state estimate: [v_C1, v_C2, i_L]
    # We don't know v_C1, so we guess 0.
    # We *do* know v_C2 and i_L from the first measurement.
    x0 = np.array([0.0, measurements[0, 0], measurements[0, 1]])
    
    # Initial Covariance (P0)
    # How sure are we of our initial state?
    # High variance for unknown v_C1, low variance for "known" v_C2, i_L.
    P0 = np.diag([1.0, 1e-4, 1e-4])

    
    print("\n--- Running Extended Kalman Filter (EKF) ---")
    x_est_ekf, P_est_ekf, innov_ekf = run_ekf(measurements, inputs, Q, R, x0, P0)
    print("EKF complete.")
    
    print("\n--- Running Unscented Kalman Filter (UKF) ---")
    x_est_ukf, P_est_ukf, innov_ukf = run_ukf(measurements, inputs, Q, R, x0, P0)
    print("UKF complete.")
    
    # --- Plot EKF Results ---
    plot_results(time, true_states[:, 0], x_est_ekf, P_est_ekf, 
                 f'EKF Estimation for $v_{C1}$ (Q={q_val:.1e}, R=[{r_val_v:.1e}, {r_val_i:.1e}])')
    
    plot_innovations_and_acf(innov_ekf, Ts, 'EKF Innovation Analysis')
    
    # --- Plot UKF Results ---
    plot_results(time, true_states[:, 0], x_est_ukf, P_est_ukf, 
                 f'UKF Estimation for $v_{C1}$ (Q={q_val:.1e}, R=[{r_val_v:.1e}, {r_val_i:.1e}])')
    
    plot_innovations_and_acf(innov_ukf, Ts, 'UKF Innovation Analysis')
    
    print("\nDisplaying plots...")
    plt.show()