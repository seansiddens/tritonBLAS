#!/usr/bin/env python3
"""
Compare tritonblas_gflops values between two CSV files.

Usage:
    python compare_gflops.py <file1.csv> <file2.csv>
"""

import csv
import sys
import argparse


def read_csv_file(filepath):
    """Read CSV file and return rows as list of dictionaries."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compare_gflops(file1_path, file2_path):
    """Compare tritonblas_gflops between two CSV files."""
    # Read both CSV files
    rows1 = read_csv_file(file1_path)
    rows2 = read_csv_file(file2_path)
    
    # Get the number of rows (excluding header)
    num_rows1 = len(rows1)
    num_rows2 = len(rows2)
    
    print(f"File 1: {file1_path} ({num_rows1} rows)")
    print(f"File 2: {file2_path} ({num_rows2} rows)")
    print("-" * 100)
    
    # Determine the number of rows to compare
    num_rows = min(num_rows1, num_rows2)
    
    if num_rows1 != num_rows2:
        print(f"Warning: Files have different number of rows. Comparing first {num_rows} rows.\n")
    
    # Print header
    print(f"{'Row':<5} {'m':<6} {'n':<6} {'k':<10} {'File1 GFLOP/s':<15} {'File2 GFLOP/s':<15} "
          f"{'Abs Diff':<15} {'Pct Delta':<15}")
    print("-" * 100)
    
    # Compare each row
    for i in range(num_rows):
        row1 = rows1[i]
        row2 = rows2[i]
        
        try:
            gflops1 = float(row1['tritonblas_gflops'])
            gflops2 = float(row2['tritonblas_gflops'])
            
            # Calculate absolute difference
            abs_diff = abs(gflops2 - gflops1)
            
            # Calculate percentage delta: ((value2 - value1) / value1) * 100
            if gflops1 != 0:
                pct_delta = ((gflops2 - gflops1) / gflops1) * 100
            else:
                pct_delta = float('inf') if gflops2 != 0 else 0.0
            
            # Get m, n, k for context
            m = row1.get('m', 'N/A')
            n = row1.get('n', 'N/A')
            k = row1.get('k', 'N/A')
            
            print(f"{i+1:<5} {m:<6} {n:<6} {k:<10} {gflops1:<15.2f} {gflops2:<15.2f} "
                  f"{abs_diff:<15.2f} {pct_delta:<15.2f}%")
            
        except (KeyError, ValueError) as e:
            print(f"Row {i+1}: Error processing row - {e}")
            continue
    
    print("-" * 100)
    
    # Summary statistics
    abs_diffs = []
    pct_deltas = []
    
    for i in range(num_rows):
        row1 = rows1[i]
        row2 = rows2[i]
        
        try:
            gflops1 = float(row1['tritonblas_gflops'])
            gflops2 = float(row2['tritonblas_gflops'])
            
            abs_diff = abs(gflops2 - gflops1)
            abs_diffs.append(abs_diff)
            
            if gflops1 != 0:
                pct_delta = ((gflops2 - gflops1) / gflops1) * 100
                pct_deltas.append(pct_delta)
        except (KeyError, ValueError):
            continue
    
    if abs_diffs:
        print(f"\nSummary Statistics:")
        print(f"  Average absolute difference: {sum(abs_diffs) / len(abs_diffs):.2f}")
        print(f"  Max absolute difference: {max(abs_diffs):.2f}")
        print(f"  Min absolute difference: {min(abs_diffs):.2f}")
    
    if pct_deltas:
        print(f"  Average percentage delta: {sum(pct_deltas) / len(pct_deltas):.2f}%")
        print(f"  Max percentage delta: {max(pct_deltas):.2f}%")
        print(f"  Min percentage delta: {min(pct_deltas):.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Compare tritonblas_gflops values between two CSV files'
    )
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    
    args = parser.parse_args()
    
    compare_gflops(args.file1, args.file2)


if __name__ == '__main__':
    main()

