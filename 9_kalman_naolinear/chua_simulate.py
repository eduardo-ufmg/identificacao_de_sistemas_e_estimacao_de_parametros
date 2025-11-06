import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# --- 1. Define Constants and System Parameters ---
C1 = 30.14e-6  # 30.14 µF
C2 = 185.6e-6  # 185.6 µF
L = 52.28      # 52.28 H
R = 1673.0     # 1673 Ω
RL = 0.0       # 0 Ω
Ga = -0.801e-3 # -0.801 mS
Gb = -0.365e-3 # -0.365 mS
E = 1.74       # 1.74 V

# --- 2. Define the ODE model ---
def chua_model(t, S, ux, uy, uz):
    """
    Defines the system of differential equations.
    
    S: State vector [vC1, vC2, iL]
    t: Time
    ux, uy, uz: Control input values (constants)
    """
    vC1, vC2, iL = S
    
    # Calculate G(vC1) based on the piecewise definition
    if abs(vC1) < E:
        G_vC1 = Ga
    else:
        G_vC1 = Gb + (Ga - Gb) * E / abs(vC1)
        
    # Compute rx, ry, rz
    rx = (C1 / (R * C2)) * ux
    ry = (1 / R) * uy
    rz = (L / (R**2 * C2)) * uz
    
    # Calculate derivatives
    dvC1_dt = (1.0/C1) * ((vC2 - vC1)/R - G_vC1 * vC1 + rx)
    dvC2_dt = (1.0/C2) * ((vC1 - vC2)/R + iL + ry)
    diL_dt = (1.0/L) * (-vC2 + RL * iL + rz)
    
    return [dvC1_dt, dvC2_dt, diL_dt]

# --- 3. Simulation Function ---
def simulate_chua_with_noise(output_filename='chua_sim.dat'):
    """
    Simulates the Chua circuit with noise and saves data in the same format as pcchua_data.dat
    """
    print("Starting simulation...")
    
    # Simulation parameters
    t_start = 0.0
    t_end = 50.0
    dt = 0.01  # Time step (approximately matching the data file)
    
    # Generate time points
    t_eval = np.arange(t_start, t_end, dt)
    
    # Initial conditions (similar to the data file)
    S0 = [-1.5, 0.6, 0.0004]
    
    # Constant control inputs (matching the data file pattern)
    ux_val = -5.882e-05
    uy_val = 3.663e-06
    uz_val = 1.961e-02
    
    # Reference values (constant, matching the data file)
    xref = 3.922e-06
    yref = 3.922e-06
    zref = 3.922e-03
    
    # Run the simulation
    print("Solving ODE...")
    sol = solve_ivp(
        chua_model,
        (t_start, t_end),
        S0,
        t_eval=t_eval,
        method='RK45',
        args=(ux_val, uy_val, uz_val)
    )
    
    if not sol.success:
        print(f"Warning: ODE solver failed. Message: {sol.message}")
        return
    
    print("Simulation complete. Adding noise...")
    
    # Extract clean simulation results
    vC1_clean = sol.y[0]
    vC2_clean = sol.y[1]
    iL_clean = sol.y[2]
    
    # Add measurement noise
    # Noise levels are chosen to be realistic for the system
    noise_std_vC1 = 0.01  # Voltage noise
    noise_std_vC2 = 0.005  # Voltage noise
    noise_std_iL = 5e-5   # Current noise
    
    np.random.seed(0)  # For reproducibility
    
    vC1_noisy = vC1_clean + np.random.normal(0, noise_std_vC1, len(vC1_clean))
    vC2_noisy = vC2_clean + np.random.normal(0, noise_std_vC2, len(vC2_clean))
    iL_noisy = iL_clean + np.random.normal(0, noise_std_iL, len(iL_clean))
    
    # Create DataFrame with the same format as pcchua_data.dat
    data = pd.DataFrame({
        'time': sol.t,
        'x': vC1_noisy,
        'y': vC2_noisy,
        'z': iL_noisy,
        'xref': np.full_like(sol.t, xref),
        'yref': np.full_like(sol.t, yref),
        'zref': np.full_like(sol.t, zref),
        'ux': np.full_like(sol.t, ux_val),
        'uy': np.full_like(sol.t, uy_val),
        'uz': np.full_like(sol.t, uz_val)
    })
    
    # Save to file with the same format
    print(f"Saving data to '{output_filename}'...")
    data.to_csv(output_filename, index=False, float_format='%.3e')
    
    print(f"Simulation complete! Data saved to '{output_filename}'")
    print(f"Generated {len(sol.t)} data points from t={t_start} to t={t_end}")
    print(f"Added Gaussian noise with std: vC1={noise_std_vC1}, vC2={noise_std_vC2}, iL={noise_std_iL}")

# --- Run the simulation ---
if __name__ == "__main__":
    simulate_chua_with_noise()