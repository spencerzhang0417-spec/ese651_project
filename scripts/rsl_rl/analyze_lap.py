import csv
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "play_metrics.csv"
num_gates_per_lap = 7

prev_gates = 0
lap_times = []
lap_start = 0.0

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ng = int(row["n_gates_passed"])
        t = float(row["time_s"])

        if ng > prev_gates and ng % num_gates_per_lap == 0:
            lap_times.append(t - lap_start)
            lap_start = t
        prev_gates = ng

print(f"Gates per lap: {num_gates_per_lap}")
print(f"Laps completed: {len(lap_times)}")
for i, lt in enumerate(lap_times):
    print(f"  Lap {i+1}: {lt:.2f}s")
if lap_times:
    print(f"  Average: {sum(lap_times)/len(lap_times):.2f}s")
    print(f"  Average x3: {3*sum(lap_times)/len(lap_times):.2f}s")
    print(f"  Total: {sum(lap_times):.2f}s")
else:
    print(f"  No laps completed. Max gates passed: {prev_gates}")