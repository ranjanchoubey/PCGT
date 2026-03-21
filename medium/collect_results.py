#!/usr/bin/env python3
"""
Utility to collect experiment results into CSV.
Usage:
    python collect_results.py --dataset cora --method pcgt --num_partitions 5 \
        --runs 5 --highest_test "83.22 ± 1.14" --final_test "81.50 ± 1.56" \
        --val_epoch 294 --notes "PCGT with METIS 5 partitions"

Or parse from a log file:
    python collect_results.py --parse logfile.txt --dataset cora --method pcgt --num_partitions 5
"""
import argparse
import csv
import os
import re

CSV_PATH = os.path.join(os.path.dirname(__file__), 'results', 'experiment_results.csv')

def parse_summary_line(text):
    """Parse the final summary line from training output.
    Example: '5 runs: Highest Train: 100.00 ± 0.00 Highest val epoch:294\nHighest Test: 83.22 ± 1.14 Final Test: 81.50 ± 1.56'
    """
    runs_match = re.search(r'(\d+)\s+runs:', text)
    val_epoch_match = re.search(r'Highest val epoch:\s*(\d+)', text)
    highest_test_match = re.search(r'Highest Test:\s*([\d.]+)\s*±\s*([\d.]+)', text)
    final_test_match = re.search(r'Final Test:\s*([\d.]+)\s*±\s*([\d.]+)', text)

    if not highest_test_match or not final_test_match:
        return None

    return {
        'runs': int(runs_match.group(1)) if runs_match else 0,
        'highest_val_epoch': int(val_epoch_match.group(1)) if val_epoch_match else 0,
        'highest_test_mean': float(highest_test_match.group(1)),
        'highest_test_std': float(highest_test_match.group(2)),
        'final_test_mean': float(final_test_match.group(1)),
        'final_test_std': float(final_test_match.group(2)),
    }


def append_result(dataset, method, num_partitions, runs, 
                  highest_test_mean, highest_test_std,
                  final_test_mean, final_test_std,
                  highest_val_epoch, notes=''):
    """Append a single result row to the CSV."""
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'method', 'num_partitions', 'runs',
                           'highest_test_mean', 'highest_test_std',
                           'final_test_mean', 'final_test_std',
                           'highest_val_epoch', 'notes'])
        writer.writerow([dataset, method, num_partitions, runs,
                        highest_test_mean, highest_test_std,
                        final_test_mean, final_test_std,
                        highest_val_epoch, notes])
    print(f"Appended: {dataset}/{method} -> {highest_test_mean}±{highest_test_std}")


def print_comparison_table():
    """Print a formatted comparison table from the CSV."""
    if not os.path.exists(CSV_PATH):
        print("No results file found.")
        return

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by dataset
    datasets = sorted(set(r['dataset'] for r in rows))
    
    print(f"\n{'='*90}")
    print(f"{'Dataset':<15} {'Method':<12} {'K':>4} {'Runs':>5} {'Highest Test':>16} {'Final Test':>16} {'ValEp':>6}")
    print(f"{'='*90}")
    
    for ds in datasets:
        ds_rows = [r for r in rows if r['dataset'] == ds]
        for r in ds_rows:
            ht = f"{float(r['highest_test_mean']):.2f}±{float(r['highest_test_std']):.2f}"
            ft = f"{float(r['final_test_mean']):.2f}±{float(r['final_test_std']):.2f}"
            print(f"{r['dataset']:<15} {r['method']:<12} {r['num_partitions']:>4} {r['runs']:>5} {ht:>16} {ft:>16} {r['highest_val_epoch']:>6}")
        print(f"{'-'*90}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse', type=str, help='Parse a log file for results')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--num_partitions', type=int, default=0)
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--show', action='store_true', help='Show comparison table')
    
    # Manual entry
    parser.add_argument('--highest_test', type=str, help='e.g. "83.22 ± 1.14"')
    parser.add_argument('--final_test', type=str, help='e.g. "81.50 ± 1.56"')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--val_epoch', type=int, default=0)

    args = parser.parse_args()

    if args.show:
        print_comparison_table()
    elif args.parse:
        with open(args.parse, 'r') as f:
            text = f.read()
        result = parse_summary_line(text)
        if result:
            append_result(args.dataset, args.method, args.num_partitions,
                        result['runs'], result['highest_test_mean'], result['highest_test_std'],
                        result['final_test_mean'], result['final_test_std'],
                        result['highest_val_epoch'], args.notes)
        else:
            print("Could not parse results from log file.")
    elif args.highest_test and args.final_test:
        ht = re.match(r'([\d.]+)\s*±\s*([\d.]+)', args.highest_test)
        ft = re.match(r'([\d.]+)\s*±\s*([\d.]+)', args.final_test)
        if ht and ft:
            append_result(args.dataset, args.method, args.num_partitions,
                        args.runs, float(ht.group(1)), float(ht.group(2)),
                        float(ft.group(1)), float(ft.group(2)),
                        args.val_epoch, args.notes)
    else:
        print_comparison_table()
