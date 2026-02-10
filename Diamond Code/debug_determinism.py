import torch
import sys
sys.path.insert(0, "S:/AI/work/VRAXION_DEV/Diamond Code")
from ring_memory_model import RingMemoryModel

# Test 1
torch.manual_seed(42)
model1 = RingMemoryModel(input_size=1, num_outputs=2)
torch.manual_seed(99)  # Different seed for input
x = torch.randn(4, 16, 1)
out1, _, _ = model1(x)

# Test 2
torch.manual_seed(42)
model2 = RingMemoryModel(input_size=1, num_outputs=2)
torch.manual_seed(99)  # Same seed for input
x2 = torch.randn(4, 16, 1)
out2, _, _ = model2(x)

print("Input x match:", torch.allclose(x, x2))
print("Output match:", torch.allclose(out1, out2, atol=1e-6))
print("Output diff:", (out1 - out2).abs().max().item())

# Check model parameters
for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    match = torch.allclose(p1, p2, atol=1e-6)
    print(f"{n1}: {match}")
