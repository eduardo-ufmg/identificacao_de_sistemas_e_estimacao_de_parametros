# I was not able to do the exercise as requested with the data file provided.
# My guess is that the data file do not match the system of equations provided exactly.
# So, this script compares the experimental data with a new simulation of the system of equations
# and provides quantitative and visual comparison.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import sys

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
    Defines the system of differential equations provided.
    
    S: State vector [vC1, vC2, iL]
    t: Time
    ux, uy, uz: Interpolation functions for control inputs
    """
    vC1, vC2, iL = S
    
    # Calculate G(vC1) based on the piecewise definition
    # G(vC1) = Ga, if |vC1(t)| < E
    # G(vC1) = Gb + (Ga - Gb)E / |vC1(t)|, otherwise
    if abs(vC1) < E:
        G_vC1 = Ga
    else:
        G_vC1 = Gb + (Ga - Gb) * E / abs(vC1)
        
    # Get interpolated control inputs for the current time t
    # and compute rx, ry, rz
    rx = (C1 / (R * C2) ) * ux(t)
    ry = ( 1 / R ) * uy(t)
    rz = ( L / (R**2 * C2) ) * uz(t)
    
    # Calculate derivatives based on the equations:
    
    # \dot{v}{C_1} = \frac{1}{C_1} \left{ \frac{v{C_2} - v_{C_1}}{R} - G(v_{C_1})v_{C_1} + r_x \right}
    dvC1_dt = (1.0/C1) * ( (vC2 - vC1)/R - G_vC1 * vC1 + rx )
    
    # \dot{v}{C_2} = \frac{1}{C_2} \left{ \frac{v{C_1} - v_{C_2}}{R} + i_L + r_y \right}
    dvC2_dt = (1.0/C2) * ( (vC1 - vC2)/R + iL + ry )
    
    # \dot{i}L = \frac{1}{L} \left{ -v{C_2} + R_L i_L + r_z \right}
    diL_dt = (1.0/L) * ( -vC2 + RL * iL + rz )
    
    return [dvC1_dt, dvC2_dt, diL_dt]

# --- 3. Main execution function ---
def load_simulate_compare(filename='pcchua_data.dat'):
    """
    Loads experimental data, runs the new simulation,
    and compares the results.
    """
    try:
        # --- Load Experimental Data ---
        print(f"Attempting to load data from '{filename}'...")
        data = pd.read_csv(filename, skipinitialspace=True)
        print(f"Loaded '{filename}' successfully.")

        # Extract experimental data ("exp" = experimental)
        # We assume x -> vC1, y -> vC2, z -> iL
        t_exp = data['time'].to_numpy()
        x_exp = data['x'].to_numpy() 
        y_exp = data['y'].to_numpy() 
        z_exp = data['z'].to_numpy() 
        
        # Extract control inputs from the file
        rx_in = data['ux'].to_numpy()
        ry_in = data['uy'].to_numpy()
        rz_in = data['uz'].to_numpy()
        
        # --- Prepare for Simulation ---
        
        # 1. Create interpolators for the control inputs.
        # This allows the ODE solver to query the input value at any time t,
        # not just at the discrete time points in the file.
        print("Creating input interpolators...")
        ux = interp1d(t_exp, rx_in, bounds_error=False, fill_value=rx_in[-1])
        uy = interp1d(t_exp, ry_in, bounds_error=False, fill_value=ry_in[-1])
        uz = interp1d(t_exp, rz_in, bounds_error=False, fill_value=rz_in[-1])
        
        # 2. Set initial conditions from the first row of data
        S0 = [x_exp[0], y_exp[0], z_exp[0]]
        
        # 3. Set time span for the solver
        t_span = (t_exp[0], t_exp[-1])
        
        print("Running ODE simulation...")
        # --- Run Simulation ---
        # We use solve_ivp to integrate the ODEs
        sol = solve_ivp(
            chua_model,                          # The function defining the system
            t_span,                              # Time interval
            S0,                                  # Initial state
            t_eval=t_exp,                        # Times to evaluate the solution at
            method='RK45',                       # Standard solver
            args=(ux, uy, uz) # Extra args for our model function
        )
        
        if not sol.success:
            print(f"Warning: ODE solver did not converge. Message: {sol.message}")
            return

        print("Simulation complete.")
        
        # Extract simulation results ("sim" = simulated)
        vC1_sim = sol.y[0]
        vC2_sim = sol.y[1]
        iL_sim = sol.y[2]
        
        # --- Quantitative Comparison (RMSE) ---
        # Calculate Root Mean Square Error between experimental and simulated data
        rmse_x = np.sqrt(mean_squared_error(x_exp, vC1_sim))
        rmse_y = np.sqrt(mean_squared_error(y_exp, vC2_sim))
        rmse_z = np.sqrt(mean_squared_error(z_exp, iL_sim))
        
        print("\n--- Quantitative Comparison (RMSE) ---")
        print(f"RMSE (x vs vC1): {rmse_x:.4e}")
        print(f"RMSE (y vs vC2): {rmse_y:.4e}")
        print(f"RMSE (z vs iL):  {rmse_z:.4e}\n")
        
        # --- Plotting Comparison ---
        print("Generating comparison plots...")
        fig = plt.figure(figsize=(14, 12))
        fig.suptitle('Experimental Data vs. Simulated Model', fontsize=16)

        # Plot 1: 3D Trajectory Comparison
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(x_exp, y_exp, z_exp, lw=0.5, label='Experimental (x,y,z)')
        ax1.plot(vC1_sim, vC2_sim, iL_sim, lw=0.5, label='Simulated (vC1,vC2,iL)', linestyle='--')
        ax1.set_title('3D Phase Portrait Comparison')
        ax1.set_xlabel('x / vC1')
        ax1.set_ylabel('y / vC2')
        ax1.set_zlabel('z / iL')
        ax1.legend()

        # Plot 2: State x/vC1 vs. Time
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_exp, x_exp, label='Experimental (x)', alpha=0.8)
        ax2.plot(t_exp, vC1_sim, label='Simulated (vC1)', linestyle='--')
        ax2.set_title('State x vs. Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('State Value')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: State y/vC2 vs. Time
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t_exp, y_exp, label='Experimental (y)', alpha=0.8)
        ax3.plot(t_exp, vC2_sim, label='Simulated (vC2)', linestyle='--')
        ax3.set_title('State y vs. Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('State Value')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: State z/iL vs. Time
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t_exp, z_exp, label='Experimental (z)', alpha=0.8)
        ax4.plot(t_exp, iL_sim, label='Simulated (iL)', linestyle='--')
        ax4.set_title('State z vs. Time')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('State Value')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        
        # Save the figure to a file
        plot_filename = "simulation_comparison.png"
        plt.savefig(plot_filename)
        print(f"Saved comparison plot to {plot_filename}")
        
        # Or display it
        # plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the file is in the same directory as the script.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# --- Run the main function ---
if __name__ == "__main__":
    load_simulate_compare()

# My conclusion is that the data file provided does not correspond exactly to the system of equations given.
# So, I will use the simulation in the exercises and inject noise as needed.
