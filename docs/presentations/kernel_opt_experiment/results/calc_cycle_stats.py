#!/usr/bin/env python3 
import sys
import os
import statistics

log_dir = sys.argv[1]

log_path = os.path.join(log_dir, "AIESimulator.log")
with open(log_path) as f:
    data = f.read()
    lines = [l for l in data.splitlines()
             if l.startswith("Loop cycle count")]
    cycles = [int(l.split(":")[1].strip()) for l in lines]

    min_ = min(cycles)
    max_ = max(cycles)
    avg = sum(cycles) / len(cycles)
    stddev = statistics.stdev(cycles)

    loop_name = os.path.dirname(log_path)
    print("| Loop | Min | Max | Avg | Std |")
    print("|------|-----|-----|-----|-----|")
    print(f"| {loop_name} | {min_} | {max_} | {avg:.1f} | {stddev:.1f} |")
