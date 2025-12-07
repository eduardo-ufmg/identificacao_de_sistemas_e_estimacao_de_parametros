#!/usr/bin/env python3
"""compute_differential_odo.py

Read an odometry CSV (default: odo.csv) and write a new file (default: odo_diff.csv)
where every row contains the difference between the current row and the previous one.
If the input file contains a header line (non-numeric first row), the header is preserved
and the first data row in the delta file will be all zeros. If no header is present,
the first output row will be all zeros as well.
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import List


def parse_row_as_floats(row: List[str]) -> List[float]:
    """Try to parse a row's cells into floats. Raise ValueError if any cell is not a float."""
    return [float(v.strip()) for v in row]


def compute_differential(in_path: str, out_path: str) -> None:
    with open(in_path, newline="") as fin:
        reader = csv.reader(fin)
        rows = [r for r in reader if r and not all(c.strip() == "" for c in r)]

    if not rows:
        print(f"Input file '{in_path}' appears empty. Nothing written.")
        return

    # detect header: if first row cannot be parsed as floats, treat as header
    has_header = False
    try:
        prev = parse_row_as_floats(rows[0])
        start_idx = 0
    except ValueError:
        has_header = True
        start_idx = 1
        if len(rows) < 2:
            print(f"Input file '{in_path}' contains only header and no numeric rows.")
            return
        prev = parse_row_as_floats(rows[1])

    # open output file and write
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)

        if has_header:
            writer.writerow(rows[0])  # preserve header

        # write first data row as zeros (same length as columns)
        zeros = [0.0] * len(prev)
        writer.writerow([format_val(v) for v in zeros])

        # iterate through remaining data rows, starting at start_idx+1
        for i in range(start_idx + 1, len(rows)):
            try:
                cur = parse_row_as_floats(rows[i])
            except ValueError:
                # Skip or warn about non-numeric rows; we'll print a warning and skip
                print(f"Warning: skipping non-numeric row {i+1} in '{in_path}': {rows[i]}")
                continue

            if len(cur) != len(prev):
                # if mismatch in columns, try to align by truncation or padding
                min_len = min(len(cur), len(prev))
                cur = cur[:min_len]
                prev = prev[:min_len]

            diff = [c - p for c, p in zip(cur, prev)]
            writer.writerow([format_val(v) for v in diff])
            prev = cur


def format_val(v: float) -> str:
    # Keep reasonable precision while avoiding scientific notation for small deltas
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute differential odometry: output changes between consecutive rows"
    )
    parser.add_argument("--input", "-i", default="odo.csv", help="Input CSV file")
    parser.add_argument("--output", "-o", default="odo_diff.csv", help="Output CSV file")
    args = parser.parse_args(argv)

    try:
        compute_differential(args.input, args.output)
        print(f"Wrote differential file to '{args.output}'.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
