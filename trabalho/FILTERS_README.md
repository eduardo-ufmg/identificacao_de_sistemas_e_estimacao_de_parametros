# Robot Localization Filters

This project implements three state estimation algorithms for robot localization:
- **Particle Filter (PF)** - Monte Carlo localization
- **Extended Kalman Filter (EKF)** - First-order linearization
- **Unscented Kalman Filter (UKF)** - Sigma point transform

## Overview

Each filter estimates the robot's pose (x, y, θ) by fusing:
- **Odometry measurements**: Differential motion increments (dx, dy, dθ)
- **Laser range measurements**: 361 laser beams covering 360° around the robot
- **Map**: Binary occupancy grid of the environment

## Files

### Core Filter Implementations
- `PF.py` - Particle Filter implementation
- `EKF.py` - Extended Kalman Filter implementation  
- `UKF.py` - Unscented Kalman Filter implementation

### Supporting Files
- `compare_filters.py` - Run all filters and generate comparison plots
- `OdometryDynamicModel.py` - Differential odometry motion model
- `show_trajectory.py` - Visualize odometry vs. reference trajectories

### Data Files
- `ref.csv` - Ground truth reference trajectory (6291 points)
- `odo_diff.csv` - Differential odometry measurements (dx, dy, dθ)
- `laser.csv` - Laser range finder data (361 readings per scan)
- `map.png` - Environment map (binary occupancy grid)
- `map_info.json` - Map metadata (resolution, bounds, initial pose)

## Usage

### Run Individual Filters

Each filter can be run independently:

```bash
# Particle Filter
python PF.py

# Extended Kalman Filter
python EKF.py

# Unscented Kalman Filter
python UKF.py
```

Each script will:
1. Load odometry and laser data
2. Run the filter over all 6291 timesteps
3. Save estimates to CSV file
4. Print error statistics

### Run Comparison

To run all filters and generate comparison plots:

```bash
python compare_filters.py
```

This will:
- Run all three filters plus baseline odometry
- Compute error statistics for each
- Generate comprehensive comparison plots
- Save results to `filter_comparison.png`

## Algorithm Details

### Particle Filter (PF.py)

**Approach**: Represents the belief as a set of weighted particles.

**Key Features**:
- 500 particles by default
- Systematic resampling when effective sample size < 50%
- Ray casting for laser likelihood computation
- Motion model with Gaussian noise

**Parameters**:
- Motion noise: σ = [0.05, 0.05, 0.02] m, m, rad
- Laser noise: σ = 0.1 m
- Uses every 10th laser beam for efficiency

**Advantages**: Non-parametric, handles multi-modal distributions
**Disadvantages**: Computationally expensive, particle depletion risk

### Extended Kalman Filter (EKF.py)

**Approach**: Linearizes nonlinear models using first-order Taylor expansion.

**Key Features**:
- Gaussian belief representation (mean + covariance)
- Numerical Jacobian computation for measurement model
- Sequential laser beam processing
- Angle normalization

**Parameters**:
- Process noise: Q = diag([0.01², 0.01², 0.005²])
- Measurement noise: R = 0.05²
- Uses every 15th laser beam

**Advantages**: Efficient, provides uncertainty estimates
**Disadvantages**: Linearization errors, assumes unimodal Gaussian

### Unscented Kalman Filter (UKF.py)

**Approach**: Uses unscented transform with sigma points to capture nonlinearities.

**Key Features**:
- 7 sigma points (2n+1 for n=3 dimensions)
- Handles nonlinear motion and measurement models
- No Jacobian computation needed
- Cholesky decomposition for sigma point generation

**Parameters**:
- α = 0.001 (sigma point spread)
- β = 2.0 (prior knowledge, optimal for Gaussian)
- κ = 0.0 (secondary scaling)
- Process noise: Q = diag([0.01², 0.01², 0.005²])
- Measurement noise: R = 0.05²

**Advantages**: Better nonlinearity handling than EKF, no Jacobians
**Disadvantages**: Still assumes Gaussian, more expensive than EKF

## Implementation Details

### Motion Model

All filters use the differential odometry model:
```
x(t+1) = x(t) + dx
y(t+1) = y(t) + dy
θ(t+1) = θ(t) + dθ
```

Where (dx, dy, dθ) are measured in the global frame.

### Measurement Model

Ray casting is used to predict laser ranges:
1. For each laser beam at angle α relative to robot
2. Cast ray from (x, y) at angle (θ + α)
3. Find first obstacle hit in occupancy grid
4. Return distance to obstacle

This is implemented efficiently with:
- Pre-computed distance transform
- Coarse ray stepping at map resolution
- Early termination at max range

### Map Representation

- Binary occupancy grid (563 × 754 pixels)
- Resolution: 0.0962 m/pixel
- Bounds: x ∈ [-5.84, 48.24], y ∈ [-30.73, 41.78]
- 1 = free space, 0 = occupied

## Tuning Tips

### Increasing Accuracy
- Increase number of particles (PF)
- Use more laser beams in update
- Reduce process/measurement noise
- Adjust UKF sigma point parameters

### Improving Speed
- Reduce number of particles (PF)
- Use fewer laser beams
- Increase ray casting step size
- Skip update steps (process every Nth measurement)

### Handling Divergence
- Increase process noise (allow more uncertainty)
- Check initial pose accuracy
- Verify map alignment
- Add particle injection (PF)

## Dependencies

```python
numpy
matplotlib
scipy
json
```

All standard scientific Python libraries.

## References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation.
- Arulampalam, M. S., et al. (2002). A tutorial on particle filters.

## Author

Eduardo - UFMG ISEP Course Project
