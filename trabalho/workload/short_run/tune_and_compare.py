"""
Unified script to tune filter parameters and then compare them across decimation levels.

This script:
1. Runs tune_filters.py to perform Bayesian optimization on the balanced dataset
2. Loads the tuned parameters
3. Runs compare_filters.py with the tuned parameters across all decimation levels
"""

import json
import time
import sys
import os

# Import the main functions from both modules
from tune_filters import main as tune_main
from compare_filters import main as compare_main


def main():
    """Main execution: tune filters, then compare with tuned parameters."""
    
    print("\n" + "="*80)
    print("INTEGRATED FILTER TUNING AND COMPARISON PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Perform Bayesian optimization tuning on the balanced dataset")
    print("  2. Load the best parameters found during tuning")
    print("  3. Run filter comparison across all decimation levels using tuned parameters")
    print("\n" + "="*80 + "\n")
    
    total_start_time = time.time()
    
    # =========================================================================
    # PHASE 1: TUNING
    # =========================================================================
    print("\n" + "#"*80)
    print("# PHASE 1: PARAMETER TUNING")
    print("#"*80)
    
    try:
        # Run tuning
        tuning_results = tune_main(save_intermediate=True)
        
        if not tuning_results or all(v.get('best_params') is None for v in tuning_results.values()):
            print("\nERROR: Tuning failed to produce valid parameters. Aborting.")
            return False
        
        # Load tuned parameters
        print("\nLoading tuned parameters from tuning_results.json...")
        with open('tuning_results.json', 'r') as f:
            tuned_params = json.load(f)
        
        print("\nTuned parameters summary:")
        for filter_name in ['PF', 'EKF', 'UKF']:
            if filter_name in tuned_params:
                print(f"  {filter_name}: Mean error = {tuned_params[filter_name]['best_score']:.4f} m")
        
    except Exception as e:
        print(f"\nERROR during tuning phase: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # PHASE 2: COMPARISON WITH TUNED PARAMETERS
    # =========================================================================
    print("\n" + "#"*80)
    print("# PHASE 2: MULTI-LEVEL COMPARISON WITH TUNED PARAMETERS")
    print("#"*80)
    
    try:
        # Run comparison with tuned parameters
        compare_main(tuned_params=tuned_params)
        
    except Exception as e:
        print(f"\nERROR during comparison phase: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("INTEGRATED PIPELINE COMPLETE")
    print("="*80)
    print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\nGenerated files:")
    print("  - tuning_result_PF.pkl, tuning_result_EKF.pkl, tuning_result_UKF.pkl")
    print("  - tuning_results.json (best parameters)")
    print("  - results_<level>_<filter>.csv (estimates for each decimation level)")
    print("  - results_<level>_statistics.txt (statistics for each level)")
    print("  - comparison_<level>.png (plots for each level)")
    print("="*80 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
