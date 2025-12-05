"""
Bayesian Grid Search Tuning for Robot Localization Filters

This script performs Bayesian optimization-based grid search tuning of the
Particle Filter (PF), Extended Kalman Filter (EKF), and Unscented Kalman Filter (UKF)
over the balanced decimation dataset using scikit-optimize (skopt).
"""

import numpy as np
import json
import time
from pathlib import Path
import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer
import warnings
warnings.filterwarnings('ignore')

from PF import ParticleFilter
from EKF import ExtendedKalmanFilter
from UKF import UnscentedKalmanFilter


# Balanced dataset configuration
DATASET = {
    'odo_file': 'odo_diff_balanced.csv',
    'ref_file': 'ref_balanced.csv',
    'laser_file': 'laser.csv'
}


# Define parameter search spaces for each filter using skopt Space
SEARCH_SPACES = {
    'PF': [
        Real(0.01, 0.2, name='motion_noise_x'),
        Real(0.01, 0.2, name='motion_noise_y'),
        Real(0.005, 0.05, name='motion_noise_theta'),
        Real(0.05, 0.3, name='laser_noise_std'),
        Real(0.3, 0.8, name='resample_threshold')
    ],
    'EKF': [
        Real(0.005, 0.05, name='Q_x'),
        Real(0.005, 0.05, name='Q_y'),
        Real(0.001, 0.02, name='Q_theta'),
        Real(0.02, 0.15, name='R_laser'),
        Integer(5, 30, name='laser_beam_skip')
    ],
    'UKF': [
        Real(1e-4, 1e-2, name='alpha'),
        Real(1.5, 2.5, name='beta'),
        Real(0.0, 1.0, name='kappa'),
        Real(0.005, 0.05, name='Q_x'),
        Real(0.005, 0.05, name='Q_y'),
        Real(0.001, 0.02, name='Q_theta'),
        Real(0.02, 0.15, name='R_laser'),
        Integer(5, 30, name='laser_beam_skip')
    ]
}


class FilterTuner:
    """Bayesian optimization-based tuner for filter parameters using skopt."""
    
    def __init__(self, filter_name, search_space):
        """
        Initialize the tuner.
        
        Args:
            filter_name: 'PF', 'EKF', or 'UKF'
            search_space: List of skopt Space dimensions
        """
        self.filter_name = filter_name
        self.search_space = search_space
        self.param_names = [dim.name for dim in search_space]
        self.iteration = 0
        self.best_score = float('inf')  # We minimize, so lower is better
        self.best_params = None
        
        # Load reference and odometry data
        self.reference = np.genfromtxt(DATASET['ref_file'], delimiter=',')
        self.odo_diff_data = np.genfromtxt(DATASET['odo_file'], delimiter=',')
        self.laser_data = np.genfromtxt(DATASET['laser_file'], delimiter=',')
        
        # Get initial pose
        with open('map_info.json', 'r') as f:
            self.map_info = json.load(f)
        self.initial_pose = np.array(self.map_info['initial_pose'])
    
    def objective(self, params_list):
        """
        Objective function to minimize (negative because we want to minimize error).
        
        Args:
            params_list: List of parameter values in order of search_space
            
        Returns:
            Mean position error (to be minimized)
        """
        self.iteration += 1
        
        # Create parameter dictionary
        params = dict(zip(self.param_names, params_list))
        
        print(f"    Iteration {self.iteration}: Testing parameters:")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.6f}")
            else:
                print(f"      {key}: {value}")
        
        try:
            if self.filter_name == 'PF':
                filter_obj = ParticleFilter(num_particles=300, config=params)
                estimates = filter_obj.run_filter(
                    odo_diff_file=DATASET['odo_file'],
                    laser_file=DATASET['laser_file'],
                    initial_pose=self.initial_pose,
                    initial_uncertainty=np.array([0.5, 0.5, 0.1])
                )
            elif self.filter_name == 'EKF':
                filter_obj = ExtendedKalmanFilter(config=params)
                estimates, _ = filter_obj.run_filter(
                    odo_diff_file=DATASET['odo_file'],
                    laser_file=DATASET['laser_file'],
                    initial_state=self.initial_pose,
                    initial_covariance=np.diag([0.1, 0.1, 0.05])**2
                )
            elif self.filter_name == 'UKF':
                filter_obj = UnscentedKalmanFilter(config=params)
                estimates, _ = filter_obj.run_filter(
                    odo_diff_file=DATASET['odo_file'],
                    laser_file=DATASET['laser_file'],
                    initial_state=self.initial_pose,
                    initial_covariance=np.diag([0.1, 0.1, 0.05])**2
                )
            else:
                return float('inf')
            
            # Ensure lengths match
            min_len = min(len(estimates), len(self.reference))
            estimates = estimates[:min_len]
            reference = self.reference[:min_len]
            
            # Compute mean position error
            position_errors = np.sqrt((reference[:, 0] - estimates[:, 0])**2 + 
                                    (reference[:, 1] - estimates[:, 1])**2)
            mean_error = np.mean(position_errors)
            
            print(f"      Mean error: {mean_error:.4f} m")
            
            # Track best
            if mean_error < self.best_score:
                self.best_score = mean_error
                self.best_params = params.copy()
                print(f"      *** NEW BEST SCORE: {mean_error:.4f} m ***")
            
            print()
            return mean_error
            
        except Exception as e:
            print(f"      Evaluation error: {e}")
            print()
            return float('inf')
    
    def tune(self, n_iterations=15):
        """
        Perform Bayesian optimization-based tuning using skopt.
        
        Args:
            n_iterations: Number of tuning iterations
            
        Returns:
            Dictionary with best parameters and result object
        """
        print(f"\n{'='*70}")
        print(f"Tuning {self.filter_name} with Bayesian Optimization (skopt)")
        print(f"{'='*70}")
        print(f"Search space dimensions: {len(self.search_space)}")
        print(f"Iterations: {n_iterations}\n")
        
        start_time = time.time()
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=self.objective,
            dimensions=self.search_space,
            acq_func='EI',  # Expected Improvement
            n_calls=n_iterations,
            n_initial_points=min(5, n_iterations),
            random_state=42,
            n_jobs=1,
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        if result is None:
            print("Tuning failed: No result returned.")
            return {
                'best_params': [],
                'best_score': None,
                'result': []
            }

        print(f"\nOptimization completed in {elapsed_time:.1f}s")
        print(f"Best score achieved: {result.fun:.4f} m")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'result': result
        }


def run_all_tuning():
    """Run tuning for all three filters."""
    results = {}
    
    for filter_name in ['PF', 'EKF', 'UKF']:
        print(f"\n{'#'*70}")
        print(f"# TUNING {filter_name}")
        print(f"{'#'*70}")
        
        tuner = FilterTuner(filter_name, SEARCH_SPACES[filter_name])
        tuning_result = tuner.tune(n_iterations=15)
        results[filter_name] = tuning_result
        
        # Save intermediate results
        with open(f'tuning_result_{filter_name}.pkl', 'wb') as f:
            pickle.dump(tuning_result, f)
        
        print(f"\nBest parameters for {filter_name}:")
        for key, value in sorted(tuning_result['best_params'].items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"Best score (mean error): {tuning_result['best_score']:.4f} m")
    
    return results


def save_results(results):
    """Save tuning results to JSON for easy reference."""
    output = {}
    
    for filter_name, result in results.items():
        output[filter_name] = {
            'best_score': float(result['best_score']),
            'best_params': {}
        }
        
        for key, value in result['best_params'].items():
            if isinstance(value, (int, np.integer)):
                output[filter_name]['best_params'][key] = int(value)
            else:
                output[filter_name]['best_params'][key] = float(value)
    
    with open('tuning_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print("TUNING RESULTS SUMMARY")
    print("="*70 + "\n")
    
    for filter_name in ['PF', 'EKF', 'UKF']:
        print(f"{filter_name}:")
        print(f"  Best mean error: {output[filter_name]['best_score']:.4f} m")
        print(f"  Best parameters:")
        for key, value in sorted(output[filter_name]['best_params'].items()):
            print(f"    {key}: {value}")
        print()


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("BAYESIAN GRID SEARCH FILTER TUNING (using scikit-optimize)")
    print("="*70)
    print(f"Dataset: Balanced decimation")
    print(f"Filters: PF, EKF, UKF\n")
    
    start_time_total = time.time()
    
    # Run tuning for all filters
    results = run_all_tuning()
    
    total_time = time.time() - start_time_total
    
    # Save results
    save_results(results)
    
    print(f"\nTotal tuning time: {total_time/60:.1f} minutes")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
