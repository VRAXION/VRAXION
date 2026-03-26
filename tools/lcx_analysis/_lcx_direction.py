"""Check if LCX cells moved in both directions or just one."""
import json
import numpy as np

path = r"S:\AI\work\VRAXION_DEV\Diamond Code\logs\swarm\matrix_history.jsonl"
entries = []
with open(path) as f:
    for line in f:
        entries.append(json.loads(line))

lcx_0 = np.array(entries[0]['lcx_after'])
lcx_499 = np.array(entries[-1]['lcx_after'])
delta = lcx_499 - lcx_0

went_up = (delta > 0.05).sum()
went_down = (delta < -0.05).sum()
stayed = 64 - went_up - went_down

print(f"Cells that went UP   (more fuchsia): {went_up}/64")
print(f"Cells that went DOWN (more cyan):    {went_down}/64")
print(f"Cells that barely moved:             {stayed}/64")
print()

# Show the actual values side by side
print("     Step 0   ->  Step 499  (delta)")
print("     -------      --------  -------")
for i in range(64):
    r, c = i // 8, i % 8
    arrow = "^^^" if delta[i] > 0.2 else ("vvv" if delta[i] < -0.2 else " . ")
    print(f"  [{r},{c}]  {lcx_0[i]:+.3f}  ->  {lcx_499[i]:+.3f}  ({delta[i]:+.3f}) {arrow}")

print()
# Top 5 biggest increases and decreases
idx_sorted = np.argsort(delta)
print("TOP 5 cells that got DARKER (more negative/cyan):")
for i in idx_sorted[:5]:
    r, c = i // 8, i % 8
    print(f"  [{r},{c}]  {lcx_0[i]:+.3f} -> {lcx_499[i]:+.3f}  (delta {delta[i]:+.3f})")

print()
print("TOP 5 cells that got BRIGHTER (more positive/fuchsia):")
for i in idx_sorted[-5:][::-1]:
    r, c = i // 8, i % 8
    print(f"  [{r},{c}]  {lcx_0[i]:+.3f} -> {lcx_499[i]:+.3f}  (delta {delta[i]:+.3f})")
