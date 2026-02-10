"""Check actual runtime head structure."""

import os
import sys
sys.path.insert(0, "S:/AI/Golden Code")

os.environ['VAR_COMPUTE_DEVICE'] = 'cpu'
os.environ['VRX_ROOT'] = "S:/AI/work/VRAXION_DEV/Golden Draft"

import torch
from vraxion.instnct.absolute_hallway import AbsoluteHallway

# Create 10-class model
model = AbsoluteHallway(input_dim=1, num_classes=10, ring_len=64, slot_dim=64)

print("HEAD STRUCTURE:")
print(f"Type: {type(model.head)}")
print(f"Class: {model.head.__class__.__name__}")
print()

print("HEAD ATTRIBUTES:")
for attr in dir(model.head):
    if not attr.startswith('_'):
        val = getattr(model.head, attr)
        if isinstance(val, (torch.nn.Module, torch.nn.Parameter, torch.Tensor)):
            print(f"  {attr}: {type(val).__name__}", end="")
            if hasattr(val, 'shape'):
                print(f" shape={tuple(val.shape)}")
            else:
                print()

print()
print("HEAD SOURCE FILE:")
import inspect
print(f"  {inspect.getfile(model.head.__class__)}")

print()
print("FORWARD METHOD:")
print(inspect.getsource(model.head.forward))
