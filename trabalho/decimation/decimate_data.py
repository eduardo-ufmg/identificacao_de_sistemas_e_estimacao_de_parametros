import pandas as pd
import os

# Define file paths
data_dir = "."
datasets = ["laser", "odo", "ref"]
strategies = ["conservative", "balanced", "aggressive"]

def read_decimation_summary(strategy):
    """Read decimation summary CSV and return decimation factors as dict."""
    summary_file = os.path.join(data_dir, f"decimation_summary_{strategy}.csv")
    df = pd.read_csv(summary_file)
    return dict(zip(df['dataset'], df['decimation_factor']))

def get_strategy_factor(strategy, factors):
    """Get the decimation factor for a strategy based on its rule."""
    factors_list = list(factors.values())
    if strategy == "conservative":
        return max(factors_list)  # Highest decimation ratio
    elif strategy == "balanced":
        return sorted(factors_list)[len(factors_list) // 2]  # Median
    elif strategy == "aggressive":
        return min(factors_list)  # Smallest decimation ratio

def decimate_dataset(dataset_name, factor):
    """Read dataset CSV and decimate by keeping every nth row."""
    input_file = os.path.join(data_dir, f"{dataset_name}.csv")
    df = pd.read_csv(input_file, header=None)
    decimated_df = df.iloc[::factor, :].reset_index(drop=True)
    return decimated_df

def save_decimated_dataset(dataset_name, decimated_df, strategy):
    """Save decimated dataset with strategy suffix."""
    output_file = os.path.join(data_dir, f"{dataset_name}_{strategy}.csv")
    decimated_df.to_csv(output_file, header=False, index=False)
    print(f"Saved: {output_file} ({len(decimated_df)} rows)")

# Process each strategy
for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Processing {strategy.upper()} strategy")
    print('='*60)
    
    # Read decimation factors for this strategy
    factors = read_decimation_summary(strategy)
    print(f"Decimation factors: {factors}")
    
    # Get the factor to apply to all datasets
    factor_to_apply = get_strategy_factor(strategy, factors)
    print(f"Factor to apply to all datasets: {factor_to_apply}")
    
    # Apply to each dataset
    for dataset in datasets:
        decimated_df = decimate_dataset(dataset, factor_to_apply)
        save_decimated_dataset(dataset, decimated_df, strategy)

print(f"\n{'='*60}")
print("Decimation complete!")
print('='*60)
