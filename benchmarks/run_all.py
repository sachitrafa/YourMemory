"""
YourMemory Benchmark Suite
---------------------------
Runs all three benchmarks and prints a combined summary.

Usage:
    python benchmarks/run_all.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks import token_efficiency, stale_memory, locomo

DIVIDER = "=" * 72

def main():
    print(DIVIDER)
    print("YOURMEMORY BENCHMARK SUITE")
    print(DIVIDER)

    print("\n\n[1/3] TOKEN EFFICIENCY\n")
    te = token_efficiency.run()

    print("\n\n[2/3] STALE MEMORY PRECISION\n")
    sm = stale_memory.run()

    print("\n\n[3/3] LoCoMo-STYLE LONG-CONTEXT RECALL\n")
    lc = locomo.run()

    print("\n\n" + DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    print(f"  Token reduction (pruned noise):       {te['token_reduction_pct']}%  ({te['pruned']}/{te['total']} memories pruned)")
    print(f"  Stale memory precision:               YourMemory {sm['ym_correct']}/{sm['total']} ({sm['yourmemory_precision']}%)  vs  Baseline {sm['bl_correct']}/{sm['total']} ({sm['baseline_precision']}%)")
    print(f"  Long-context recall@5 (30 days):      YourMemory {lc['ym_recall_pct']}%  vs  Baseline {lc['bl_recall_pct']}%")
    print()
    print("Key insight:")
    print("  Baseline recall is higher overall (surfaces everything, including stale noise).")
    print("  YourMemory trades raw recall for precision: important facts persist,")
    print("  ephemeral noise decays, and outdated facts are automatically demoted.")
    print("  In a real assistant context, injecting stale/irrelevant memories wastes")
    print("  tokens and confuses the model. YourMemory only keeps what matters.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
