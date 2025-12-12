#!/usr/bin/env python3
"""
Compare Experiment 8 results: Original (boundary oscillation bug) vs Fixed solver

Run this script after re-running notebook 06_network_fragmentation.ipynb
with the fixed solver.

Usage:
    python compare_exp8_results.py

The script expects:
    - Original results backed up to: /workspace/data/experiment8_fragmentation_results_original.csv
    - New results at: /workspace/data/experiment8_fragmentation_results.csv
"""

import pandas as pd
import numpy as np
import sys

def load_results():
    """Load original and new results."""
    try:
        # Try to load original (you may need to back this up before re-running)
        original = pd.read_csv('/workspace/data/experiment8_fragmentation_results_original.csv')
        print("✓ Loaded original results")
    except FileNotFoundError:
        print("✗ Original results not found at experiment8_fragmentation_results_original.csv")
        print("  Please backup the original file before re-running the experiment.")
        original = None

    try:
        new = pd.read_csv('/workspace/data/experiment8_fragmentation_results.csv')
        print("✓ Loaded new results")
    except FileNotFoundError:
        print("✗ New results not found. Has the experiment completed?")
        new = None

    return original, new


def compare_results(original, new):
    """Generate comparison between original and fixed results."""

    print("\n" + "=" * 70)
    print("EXPERIMENT 8: ORIGINAL vs FIXED SOLVER COMPARISON")
    print("=" * 70)

    # Overall statistics
    print("\n### OVERALL STATISTICS ###\n")

    metrics = ['avg_entropy', 'avg_tip_events', 'avg_recovery_events',
               'tip_recovery_ratio', 'avg_pct_tipped']

    print(f"{'Metric':<25} {'Original':>15} {'Fixed':>15} {'Change':>15}")
    print("-" * 70)

    for metric in metrics:
        if metric in original.columns and metric in new.columns:
            orig_val = original[metric].mean()
            new_val = new[metric].mean()

            if orig_val != 0 and not np.isnan(orig_val):
                change = (new_val - orig_val) / abs(orig_val) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"

            print(f"{metric:<25} {orig_val:>15.2f} {new_val:>15.2f} {change_str:>15}")

    # By retention level
    print("\n\n### BY RETENTION LEVEL ###\n")

    print("TIP/RECOVERY RATIO:")
    print(f"{'Retention':<12} {'Original':>12} {'Fixed':>12} {'Interpretation':>30}")
    print("-" * 70)

    for retention in sorted(original['retention'].unique(), reverse=True):
        orig_ratio = original[original['retention'] == retention]['tip_recovery_ratio'].mean()
        new_ratio = new[new['retention'] == retention]['tip_recovery_ratio'].mean()

        # Interpretation
        if np.isnan(new_ratio):
            interp = "No data"
        elif new_ratio < 0.9:
            interp = "Recovery favored"
        elif new_ratio > 1.1:
            interp = "Tipping favored (ASYMMETRIC)"
        else:
            interp = "Balanced (~symmetric)"

        print(f"{retention:>10.0%} {orig_ratio:>12.3f} {new_ratio:>12.3f} {interp:>30}")

    # Event counts comparison
    print("\n\nTIP EVENTS (per run average):")
    print(f"{'Retention':<12} {'Original':>12} {'Fixed':>12} {'Reduction':>15}")
    print("-" * 55)

    for retention in sorted(original['retention'].unique(), reverse=True):
        orig_tips = original[original['retention'] == retention]['avg_tip_events'].mean()
        new_tips = new[new['retention'] == retention]['avg_tip_events'].mean()
        reduction = (1 - new_tips / orig_tips) * 100 if orig_tips > 0 else 0
        print(f"{retention:>10.0%} {orig_tips:>12.0f} {new_tips:>12.0f} {reduction:>14.1f}%")

    # By fragmentation method
    print("\n\n### BY FRAGMENTATION METHOD ###\n")

    for method in original['method'].unique():
        print(f"\n{method.upper()}:")
        orig_method = original[original['method'] == method]
        new_method = new[new['method'] == method]

        print(f"  Original: ratio={orig_method['tip_recovery_ratio'].mean():.3f}, "
              f"entropy={orig_method['avg_entropy'].mean():.0f}, "
              f"tips={orig_method['avg_tip_events'].mean():.0f}")
        print(f"  Fixed:    ratio={new_method['tip_recovery_ratio'].mean():.3f}, "
              f"entropy={new_method['avg_entropy'].mean():.0f}, "
              f"tips={new_method['avg_tip_events'].mean():.0f}")

    # Key findings
    print("\n\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    orig_event_total = original['avg_tip_events'].mean() + original['avg_recovery_events'].mean()
    new_event_total = new['avg_tip_events'].mean() + new['avg_recovery_events'].mean()
    event_reduction = (1 - new_event_total / orig_event_total) * 100

    print(f"""
1. EVENT COUNT REDUCTION: {event_reduction:.0f}%
   - Original: ~{orig_event_total:.0f} events/run (inflated by boundary oscillations)
   - Fixed: ~{new_event_total:.0f} events/run (real state transitions)

2. TIP/RECOVERY RATIO:
   - Original: {original['tip_recovery_ratio'].mean():.3f} (appeared balanced due to oscillation)
   - Fixed: {new['tip_recovery_ratio'].mean():.3f} (true dynamics)

3. ENTROPY PRODUCTION:
   - Original: {original['avg_entropy'].mean():.0f} (inflated)
   - Fixed: {new['avg_entropy'].mean():.0f}

4. BOUNDARY ISSUE IMPACT:
   The original results were dominated by numerical artifacts from cells
   oscillating between ±10 clamp boundaries. The fixed solver keeps cells
   in the bistable region (|x| < 2), revealing true tipping dynamics.
""")

    # Revised conclusions
    print("\n" + "=" * 70)
    print("REVISED CONCLUSIONS FOR PHASE 4 DOCUMENTATION")
    print("=" * 70)

    new_ratio_at_10pct = new[new['retention'] == 0.10]['tip_recovery_ratio'].mean()
    new_ratio_at_100pct = new[new['retention'] == 1.0]['tip_recovery_ratio'].mean()

    if new_ratio_at_10pct > 1.2:
        fragmentation_effect = "Fragmentation DOES create asymmetry - recovery becomes harder"
    elif new_ratio_at_10pct < 0.8:
        fragmentation_effect = "Fragmentation favors recovery - isolated cells tip back more easily"
    else:
        fragmentation_effect = "Fragmentation has minimal effect on tip/recovery balance"

    print(f"""
ORIGINAL CONCLUSION (with bug):
  "Network fragmentation reduces cascade activity but maintains thermodynamic
   balance. The tip/recovery ratio stays ~1.0 regardless of connectivity."

REVISED CONCLUSION (fixed solver):
  Ratio at 100% retention: {new_ratio_at_100pct:.3f}
  Ratio at 10% retention: {new_ratio_at_10pct:.3f}

  {fragmentation_effect}

  [Update docs/phase4_results.md with these findings]
""")


def analyze_new_only(new):
    """Analyze new results if original not available."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: FIXED SOLVER RESULTS")
    print("=" * 70)

    print("\n### SUMMARY BY RETENTION LEVEL ###\n")

    summary = new.groupby('retention').agg({
        'n_edges': 'mean',
        'avg_entropy': 'mean',
        'avg_tip_events': 'mean',
        'avg_recovery_events': 'mean',
        'tip_recovery_ratio': 'mean',
        'avg_pct_tipped': 'mean'
    }).round(2)

    print(summary.to_string())

    print("\n\n### KEY METRICS ###\n")
    print(f"Tip/Recovery Ratio range: {new['tip_recovery_ratio'].min():.3f} - {new['tip_recovery_ratio'].max():.3f}")
    print(f"Mean events per run: {new['avg_tip_events'].mean():.0f} tips, {new['avg_recovery_events'].mean():.0f} recoveries")
    print(f"% time tipped: {new['avg_pct_tipped'].mean():.1f}%")


def main():
    print("Experiment 8 Results Comparison Script")
    print("=" * 70)

    original, new = load_results()

    if original is not None and new is not None:
        compare_results(original, new)
    elif new is not None:
        print("\nNo original results to compare. Analyzing new results only:")
        analyze_new_only(new)
    else:
        print("\nNo results available to analyze.")
        sys.exit(1)


if __name__ == '__main__':
    main()
