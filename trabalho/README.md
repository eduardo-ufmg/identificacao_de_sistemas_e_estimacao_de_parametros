# Nonlinear State Estimation with UKF

This assignment estimates the robot trajectory using an Unscented Kalman Filter (UKF) with a differential-drive motion model and a laser range measurement model based on the provided occupancy grid map `icex_2.pgm`.

## Files
- `loadData.m`: Loads `data.dat`, skipping the first 5 header lines, and returns a numeric matrix with odometry (cols 1–3), reference estimate (cols 4–6), and laser readings (cols 7–367).
- `laserMap.m`: Provided helper for map and laser simulation.
- `runUKF.m`: Runs the UKF over the dataset and plots the estimated trajectory versus the reference.

## Models
- **Motion**: Unicycle/differential-drive: 
  x_{k+1} = x_k + v_k dt cos(θ_k + 0.5 w_k dt)
  y_{k+1} = y_k + v_k dt sin(θ_k + 0.5 w_k dt)
  θ_{k+1} = wrapToPi(θ_k + w_k dt)

  Velocities `v_k` and `w_k` are derived from odometry differences per sample (dt = 0.1 s).

- **Measurement**: Laser ranges simulated from the map using `laserMap.simLaser` between −90° and +90°, decimating beams (every 5th) to reduce computation. Measurement vector is the decimated ranges. The innovation compares simulated ranges to actual LIDAR readings.

## Filter and Tuning
- Filter: `unscentedKalmanFilter` (Sensor Fusion and Tracking Toolbox).
- Sigma point params: Alpha = 1e−2, Beta = 2, Kappa = 0.
- Process noise: diag([0.05^2, 0.05^2, (1°)^2]).
- Measurement noise: 0.08 m standard deviation per beam.

These values are conservative defaults; they can be refined by comparing residuals and trajectory quality.

## Run
Open MATLAB in this folder and run:
```matlab
runUKF
```
This will load `data.dat`, perform filtering, and display a plot with the map, the provided reference trajectory (red), and the UKF estimate (blue).

## Notes
- If runtime is high, increase `beamDecim` in `runUKF.m` (e.g., 7 or 9) or reduce `maxRange` slightly to cut ray marching length.
- Ensure Sensor Fusion and Tracking Toolbox is available for `unscentedKalmanFilter`.
