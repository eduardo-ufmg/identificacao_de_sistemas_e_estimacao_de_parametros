import argparse
import os

import numpy as np


def trim_file(input_path: str, start_len: int, final_len: int, output_path: str | None = None):
    """
    Trim the first start_len and last final_len rows from a CSV file.
    
    Args:
        input_path: Path to input CSV file
        start_len: Number of rows to remove from start
        final_len: Number of rows to remove from end
        output_path: Path to output CSV file (optional, defaults to {name}_trimmed.csv)
    """
    # Load data
    data = np.loadtxt(input_path, delimiter=",")
    
    # Handle 1D arrays
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_rows = len(data)
    
    # Validate trim lengths
    if start_len + final_len >= n_rows:
        raise ValueError(
            f"Cannot trim {start_len} + {final_len} = {start_len + final_len} rows "
            f"from file with only {n_rows} rows"
        )
    
    # Trim data
    if final_len > 0:
        trimmed_data = data[start_len:-final_len]
    else:
        trimmed_data = data[start_len:]
    
    # Generate output path if not provided
    if output_path is None:
        base_dir = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(base_dir, f"{name}_trimmed{ext}")
    
    # Save trimmed data
    np.savetxt(output_path, trimmed_data, delimiter=",", fmt="%.6f")
    
    print(f"Trimmed {input_path}: {n_rows} -> {len(trimmed_data)} rows")
    print(f"  Removed {start_len} from start, {final_len} from end")
    print(f"  Saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Trim the first and last rows from CSV data files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input CSV files to trim",
    )
    parser.add_argument(
        "--start-len",
        type=int,
        default=0,
        help="Number of rows to remove from start (default: 0)",
    )
    parser.add_argument(
        "--final-len",
        type=int,
        default=0,
        help="Number of rows to remove from end (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for trimmed files (default: same as input)",
    )
    args = parser.parse_args()
    
    if args.start_len < 0 or args.final_len < 0:
        parser.error("--start-len and --final-len must be non-negative")
    
    if args.start_len == 0 and args.final_len == 0:
        parser.error("At least one of --start-len or --final-len must be > 0")
    
    print(f"Trimming {len(args.files)} file(s)...")
    print(f"Start trim: {args.start_len} rows")
    print(f"Final trim: {args.final_len} rows")
    print()
    
    for input_path in args.files:
        try:
            # Generate output path
            if args.output_dir:
                base_name = os.path.basename(input_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(args.output_dir, f"{name}_trimmed{ext}")
            else:
                output_path = None
            
            trim_file(input_path, args.start_len, args.final_len, output_path)
            print()
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            print()
    
    print("Done!")


if __name__ == "__main__":
    main()
