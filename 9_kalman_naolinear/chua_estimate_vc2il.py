import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.integrate import solve_ivp

# --- 1. Load Data ---
data = pd.read_csv("chua_sim.dat")
t = data["time"].to_numpy()
dt = np.mean(np.diff(t))

# Only vC1 is measured now
z_meas = data[["x"]].to_numpy()  # 'x' is vC1

# Control inputs (constant in this simulation)
ux = data["ux"].to_numpy()
uy = data["uy"].to_numpy()
uz = data["uz"].to_numpy()

# --- 2. Chua System Parameters ---
C1 = 30.14e-6
C2 = 185.6e-6
L = 52.28
R = 1673.0
RL = 0.0
Ga = -0.801e-3
Gb = -0.365e-3
E = 1.74


# --- 3. Chua Model Functions ---
def chua_g(vC1):
    # Piecewise nonlinearity
    if np.abs(vC1) < E:
        return Ga
    else:
        return Gb + (Ga - Gb) * E / np.abs(vC1)


def chua_f(x, u):
    vC1, vC2, iL = x
    ux, uy, uz = u
    G_vC1 = chua_g(vC1)
    rx = (C1 / (R * C2)) * ux
    ry = (1 / R) * uy
    rz = (L / (R**2 * C2)) * uz
    dvC1 = (1.0 / C1) * ((vC2 - vC1) / R - G_vC1 * vC1 + rx)
    dvC2 = (1.0 / C2) * ((vC1 - vC2) / R + iL + ry)
    diL = (1.0 / L) * (-vC2 + RL * iL + rz)
    return np.array([dvC1, dvC2, diL])


def chua_F_jacobian(x, u):
    vC1, vC2, iL = x
    ux, uy, uz = u
    # Partial derivatives for EKF
    if np.abs(vC1) < E:
        dGdv = 0
        G_vC1 = Ga
    else:
        dGdv = -(Ga - Gb) * E / (np.abs(vC1) ** 2) * np.sign(vC1)
        G_vC1 = Gb + (Ga - Gb) * E / np.abs(vC1)
    # df/dx
    F = np.zeros((3, 3))
    # dvC1/dvC1
    F[0, 0] = (1.0 / C1) * (-1.0 / R - G_vC1 - vC1 * dGdv)
    # dvC1/dvC2
    F[0, 1] = (1.0 / C1) * (1.0 / R)
    # dvC1/diL
    F[0, 2] = 0
    # dvC2/dvC1
    F[1, 0] = (1.0 / C2) * (1.0 / R)
    # dvC2/dvC2
    F[1, 1] = (1.0 / C2) * (-1.0 / R)
    # dvC2/diL
    F[1, 2] = 1.0 / C2
    # diL/dvC1
    F[2, 0] = 0
    # diL/dvC2
    F[2, 1] = -1.0 / L
    # diL/diL
    F[2, 2] = (1.0 / L) * RL
    return F


def fx_ekf(x, dt, u):
    # Integrate one step using Euler
    return x + chua_f(x, u) * dt


def hx_ekf(x):
    # Measurement function: only vC1
    return np.array([x[0]])


def HJacobian_ekf(x):
    # Measurement Jacobian: dh/dx
    H = np.array([[1, 0, 0]])  # dvC1/d[vC1, vC2, iL]
    return H


def fx_ukf(x, dt, u):
    return x + chua_f(x, u) * dt


def hx_ukf(x):
    return np.array([x[0]])


# --- 4. EKF Setup ---
ekf = EKF(dim_x=3, dim_z=1)
ekf.x = np.array(
    [z_meas[0, 0], 0.0, 0.0]
)  # Initial guess: vC1 from measurement, vC2/iL unknown
ekf.P *= 1e-2
ekf.R = np.diag([0.005**2])  # Measurement noise (only vC1)
ekf.Q = np.diag([1e-7, 1e-7, 1e-9])  # Process noise

# --- 5. UKF Setup ---
points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=0)
ukf = UKF(dim_x=3, dim_z=1, fx=fx_ukf, hx=hx_ukf, dt=dt, points=points)
ukf.x = np.array([z_meas[0, 0], 0.0, 0.0])
ukf.P *= 1e-2
ukf.R = np.diag([0.005**2])
ukf.Q = np.diag([1e-7, 1e-7, 1e-9])

# --- 6. Run Filters ---
ekf_vC2 = []
ekf_iL = []
ukf_vC2 = []
ukf_iL = []

for k in range(len(t)):
    u_k = [ux[k], uy[k], uz[k]]
    # EKF predict
    F = chua_F_jacobian(ekf.x, u_k)
    ekf.F = np.eye(3) + F * dt  # Discrete-time state transition matrix
    ekf.predict()
    ekf.update(z_meas[k], HJacobian_ekf, hx_ekf)
    ekf_vC2.append(ekf.x[1])
    ekf_iL.append(ekf.x[2])
    # UKF predict
    ukf.predict(dt=dt, u=u_k)
    ukf.update(z_meas[k])
    ukf_vC2.append(ukf.x[1])
    ukf_iL.append(ukf.x[2])

ekf_vC2 = np.array(ekf_vC2)
ekf_iL = np.array(ekf_iL)
ukf_vC2 = np.array(ukf_vC2)
ukf_iL = np.array(ukf_iL)

# --- 7. Save Results ---
out = pd.DataFrame(
    {
        "time": t,
        "vC2_true": data["y"],
        "iL_true": data["z"],
        "vC2_EKF": ekf_vC2,
        "iL_EKF": ekf_iL,
        "vC2_UKF": ukf_vC2,
        "iL_UKF": ukf_iL,
        "vC1_meas": data["x"],
    }
)

out.to_csv("chua_estimate_out.csv", index=False)
print("Estimation complete. Results saved to 'chua_estimate_out.csv'.")

# --- 8. Plot and compare results ---
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, data["y"], label="vC2 Real")
plt.plot(t, ekf_vC2, label="vC2 EKF")
plt.plot(t, ukf_vC2, label="vC2 UKF")
plt.xlabel("Tempo [s]")
plt.ylabel("$v_{C2}$ [V]")
plt.title("Estimativa do Estado $v_{C2}$ do Circuito de Chua")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, data["z"], label="iL Real")
plt.plot(t, ekf_iL, label="iL EKF")
plt.plot(t, ukf_iL, label="iL UKF")
plt.xlabel("Tempo [s]")
plt.ylabel("$i_{L}$ [A]")
plt.title("Estimativa do Estado $i_{L}$ do Circuito de Chua")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("chua_comparacao_vC2_iL.png")
plt.show()
