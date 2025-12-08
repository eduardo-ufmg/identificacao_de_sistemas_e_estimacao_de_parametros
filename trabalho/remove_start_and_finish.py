import argparse
import json
import os

import numpy as np


def trim_file(
    input_path: str, start_len: int, final_len: int, output_path: str | None = None
):
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

    return output_path, trimmed_data


def update_map_info(map_info_path: str, new_initial_pose: np.ndarray):
    """
    Update the initial_pose in map_info.json file.

    Args:
        map_info_path: Path to map_info.json file
        new_initial_pose: New initial pose [x, y, theta]
    """
    try:
        with open(map_info_path, "r") as f:
            map_info = json.load(f)

        old_pose = map_info.get("initial_pose", None)
        map_info["initial_pose"] = [float(x) for x in new_initial_pose]

        with open(map_info_path, "w") as f:
            json.dump(map_info, f, indent=4)

        print(f"Updated {map_info_path}:")
        if old_pose:
            print(f"  Old initial pose: {old_pose}")
        print(f"  New initial pose: {map_info['initial_pose']}")

        return True
    except Exception as e:
        print(f"Error updating {map_info_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Trim the first and last rows from CSV data files and update map_info.json"
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
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Reference trajectory file to use for updating initial pose in map_info.json",
    )
    parser.add_argument(
        "--map-info",
        default="map_info.json",
        help="Path to map_info.json file to update (default: map_info.json)",
    )
    parser.add_argument(
        "--update-map-info",
        action="store_true",
        help="Update map_info.json with initial pose from trimmed reference file",
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

    trimmed_reference_data = None

    for input_path in args.files:
        try:
            # Generate output path
            if args.output_dir:
                base_name = os.path.basename(input_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(args.output_dir, f"{name}_trimmed{ext}")
            else:
                output_path = None

            output_file, trimmed_data = trim_file(
                input_path, args.start_len, args.final_len, output_path
            )

            # Check if this is the reference file
            if args.reference_file and os.path.abspath(input_path) == os.path.abspath(
                args.reference_file
            ):
                trimmed_reference_data = trimmed_data
                print(f"  -> Marked as reference file for map_info update")

            print()

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            print()

    # Update map_info.json if requested
    if args.update_map_info:
        if trimmed_reference_data is None:
            print(
                "Warning: --update-map-info specified but reference file not found in trimmed files"
            )
            print(f"         Expected reference file: {args.reference_file}")
            print("         Skipping map_info.json update")
        else:
            print()
            # Get first row of trimmed reference (should be [x, y, theta])
            if trimmed_reference_data.ndim == 1:
                new_initial_pose = trimmed_reference_data
            else:
                new_initial_pose = trimmed_reference_data[0]

            if len(new_initial_pose) >= 3:
                update_map_info(args.map_info, new_initial_pose[:3])
            else:
                print(
                    f"Error: Reference file has {len(new_initial_pose)} columns, expected at least 3 (x, y, theta)"
                )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
