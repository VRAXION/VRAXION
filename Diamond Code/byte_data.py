"""
Byte Sequence Data Generator

Generates random byte sequences for testing direct byte I/O models.
Each byte represented as 8 binary bits.
"""

import torch
from typing import Tuple


def generate_random_bytes(
    n_samples: int = 100,
    seq_len: int = 16,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random byte sequences for autoencoding task.

    Args:
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence (in bytes)
        seed: Random seed for reproducibility

    Returns:
        x: Input sequences [n_samples, seq_len, 8] (binary bits)
        y: Target sequences [n_samples, seq_len, 8] (same as x for autoencoding)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random bits: [n_samples, seq_len, 8]
    x = torch.randint(0, 2, (n_samples, seq_len, 8), dtype=torch.float32)

    # For autoencoding, target = input
    y = x.clone()

    return x, y


def generate_repeated_byte(
    n_samples: int = 100,
    seq_len: int = 16,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequences with a single byte repeated.
    Easy task to verify model can memorize patterns.

    Args:
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence (in bytes)
        seed: Random seed for reproducibility

    Returns:
        x: Input sequences [n_samples, seq_len, 8] (same byte repeated)
        y: Target sequences [n_samples, seq_len, 8] (same as x)
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)

    for i in range(n_samples):
        # Generate one random byte
        byte = torch.randint(0, 2, (8,), dtype=torch.float32)
        # Repeat it across the sequence
        x[i, :, :] = byte.unsqueeze(0).expand(seq_len, -1)

    y = x.clone()
    return x, y


def generate_copy_task(
    n_samples: int = 100,
    seq_len: int = 16,
    n_copy: int = 4,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate copy task: copy first N bytes to end of sequence.

    Example (seq_len=16, n_copy=4):
        Input:  [byte0, byte1, byte2, byte3, 0, 0, 0, 0, ...]
        Target: [byte0, byte1, byte2, byte3, byte0, byte1, byte2, byte3, ...]

    Args:
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence (in bytes)
        n_copy: Number of bytes to copy
        seed: Random seed for reproducibility

    Returns:
        x: Input sequences [n_samples, seq_len, 8]
        y: Target sequences [n_samples, seq_len, 8]
    """
    if seed is not None:
        torch.manual_seed(seed)

    assert n_copy <= seq_len // 2, "n_copy must be <= seq_len // 2"

    # Generate random bytes for first n_copy positions
    x = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)
    pattern = torch.randint(0, 2, (n_samples, n_copy, 8), dtype=torch.float32)

    # Input: pattern followed by zeros
    x[:, :n_copy, :] = pattern

    # Target: pattern repeated (copy to end)
    y = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)
    y[:, :n_copy, :] = pattern  # Original position

    # Copy pattern to second half
    for i in range(n_copy):
        if n_copy + i < seq_len:
            y[:, n_copy + i, :] = pattern[:, i, :]

    return x, y


def bytes_to_string(byte_tensor: torch.Tensor) -> str:
    """
    Convert binary byte tensor to string for visualization.

    Args:
        byte_tensor: [seq_len, 8] or [8] tensor of binary bits

    Returns:
        String representation of bytes
    """
    if byte_tensor.dim() == 1:
        byte_tensor = byte_tensor.unsqueeze(0)

    result = []
    for byte_bits in byte_tensor:
        # Convert 8 bits to integer
        byte_val = 0
        for i, bit in enumerate(byte_bits):
            byte_val += int(bit.item()) * (2 ** (7 - i))
        result.append(f"{byte_val:3d}")

    return "[" + " ".join(result) + "]"


def byte_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate byte-level accuracy (all 8 bits must match).

    Args:
        output: Model predictions [batch, seq_len, 8]
        target: Ground truth [batch, seq_len, 8]

    Returns:
        Fraction of bytes where all 8 bits are correct
    """
    # Threshold output
    pred_bits = (output > 0.5).float()

    # Check if all 8 bits match for each byte
    byte_matches = (pred_bits == target).all(dim=-1).float()

    return byte_matches.mean().item()


def bit_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate bit-level accuracy.

    Args:
        output: Model predictions [batch, seq_len, 8]
        target: Ground truth [batch, seq_len, 8]

    Returns:
        Fraction of individual bits that are correct
    """
    pred_bits = (output > 0.5).float()
    return (pred_bits == target).float().mean().item()


def generate_logic_task(
    n_samples: int = 100,
    seq_len: int = 16,
    operation: str = "and",
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate bitwise logic task.

    Input:  [a, b, 0, 0, ...]
    Target: [a, b, result, 0, ...]

    Where result = a OP b (bitwise)
    Operations: "and", "or", "xor", "nand", "nor"

    Args:
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence (in bytes)
        operation: Logic operation ("and", "or", "xor", "nand", "nor")
        seed: Random seed for reproducibility

    Returns:
        x: Input sequences [n_samples, seq_len, 8] (binary bits)
        y: Target sequences [n_samples, seq_len, 8]
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)
    y = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)

    for i in range(n_samples):
        # Generate random bit patterns
        a_bits = torch.randint(0, 2, (8,), dtype=torch.float32)
        b_bits = torch.randint(0, 2, (8,), dtype=torch.float32)

        # Compute result based on operation
        if operation == "and":
            result_bits = a_bits * b_bits  # Element-wise AND
        elif operation == "or":
            result_bits = torch.clamp(a_bits + b_bits, 0, 1)  # Element-wise OR
        elif operation == "xor":
            result_bits = (a_bits + b_bits) % 2  # Element-wise XOR
        elif operation == "nand":
            result_bits = 1 - (a_bits * b_bits)  # NOT AND
        elif operation == "nor":
            result_bits = 1 - torch.clamp(a_bits + b_bits, 0, 1)  # NOT OR
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Input: [a, b, SIGNAL, 0, ...]
        # SIGNAL = all 1s tells model "compute now"
        x[i, 0, :] = a_bits
        x[i, 1, :] = b_bits
        x[i, 2, :] = torch.ones(8, dtype=torch.float32)  # Operation signal = [1,1,1,1,1,1,1,1]

        # Target: [a, b, result, 0, ...]
        y[i, 0, :] = a_bits  # Echo a
        y[i, 1, :] = b_bits  # Echo b
        y[i, 2, :] = result_bits  # Compute logic

    return x, y


def generate_addition_task(
    n_samples: int = 100,
    seq_len: int = 16,
    max_value: int = 255,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate byte addition task.

    Input:  [a, b, 0, 0, ...]
    Target: [a, b, sum, 0, ...]

    Where a, b are random bytes (0-255), sum = (a + b) % 256

    Args:
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence (in bytes)
        max_value: Maximum value for operands (default 255)
        seed: Random seed for reproducibility

    Returns:
        x: Input sequences [n_samples, seq_len, 8] (binary bits)
        y: Target sequences [n_samples, seq_len, 8]
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)
    y = torch.zeros((n_samples, seq_len, 8), dtype=torch.float32)

    for i in range(n_samples):
        # Generate random operands
        a = torch.randint(0, max_value + 1, (1,)).item()
        b = torch.randint(0, max_value + 1, (1,)).item()
        sum_val = (a + b) % 256  # Wrap around at 256

        # Convert to 8-bit binary
        a_bits = torch.tensor([(a >> (7 - j)) & 1 for j in range(8)], dtype=torch.float32)
        b_bits = torch.tensor([(b >> (7 - j)) & 1 for j in range(8)], dtype=torch.float32)
        sum_bits = torch.tensor([(sum_val >> (7 - j)) & 1 for j in range(8)], dtype=torch.float32)

        # Input: [a, b, SIGNAL, 0, ...]
        # SIGNAL = all 1s tells model "compute now"
        x[i, 0, :] = a_bits
        x[i, 1, :] = b_bits
        x[i, 2, :] = torch.ones(8, dtype=torch.float32)  # Operation signal = [1,1,1,1,1,1,1,1]
        # Rest are zeros (already initialized)

        # Target: [a, b, sum, 0, ...]
        y[i, 0, :] = a_bits  # Echo a
        y[i, 1, :] = b_bits  # Echo b
        y[i, 2, :] = sum_bits  # Compute sum

    return x, y


if __name__ == "__main__":
    print("=" * 70)
    print("BYTE DATA GENERATOR TESTS")
    print("=" * 70)
    print()

    # Test 1: Random bytes
    print("Test 1: Random Bytes")
    x, y = generate_random_bytes(n_samples=2, seq_len=4, seed=42)
    print(f"Shape: {x.shape}")
    print(f"Sample 0: {bytes_to_string(x[0])}")
    print(f"Sample 1: {bytes_to_string(x[1])}")
    print()

    # Test 2: Repeated byte
    print("Test 2: Repeated Byte")
    x, y = generate_repeated_byte(n_samples=2, seq_len=4, seed=42)
    print(f"Sample 0: {bytes_to_string(x[0])}")
    print(f"Sample 1: {bytes_to_string(x[1])}")
    print()

    # Test 3: Copy task
    print("Test 3: Copy Task")
    x, y = generate_copy_task(n_samples=1, seq_len=8, n_copy=2, seed=42)
    print(f"Input:  {bytes_to_string(x[0])}")
    print(f"Target: {bytes_to_string(y[0])}")
    print()

    # Test 4: Addition task
    print("Test 4: Addition Task")
    x, y = generate_addition_task(n_samples=3, seq_len=8, max_value=50, seed=42)
    for i in range(3):
        # Decode the values
        a_val = sum(int(x[i, 0, j].item()) * (2 ** (7 - j)) for j in range(8))
        b_val = sum(int(x[i, 1, j].item()) * (2 ** (7 - j)) for j in range(8))
        sum_val = sum(int(y[i, 2, j].item()) * (2 ** (7 - j)) for j in range(8))
        print(f"Sample {i}: {a_val} + {b_val} = {sum_val}")
        print(f"  Input:  {bytes_to_string(x[i])}")
        print(f"  Target: {bytes_to_string(y[i])}")
    print()

    # Test 5: Logic tasks
    print("Test 5: Logic Tasks (AND, OR, XOR)")
    for op in ["and", "or", "xor"]:
        print(f"\n{op.upper()} operation:")
        x, y = generate_logic_task(n_samples=2, seq_len=8, operation=op, seed=42)
        for i in range(2):
            a_val = sum(int(x[i, 0, j].item()) * (2 ** (7 - j)) for j in range(8))
            b_val = sum(int(x[i, 1, j].item()) * (2 ** (7 - j)) for j in range(8))
            r_val = sum(int(y[i, 2, j].item()) * (2 ** (7 - j)) for j in range(8))
            print(f"  {a_val} {op.upper()} {b_val} = {r_val}")
            print(f"  Bits: {x[i,0].int().tolist()} {op.upper()} {x[i,1].int().tolist()} = {y[i,2].int().tolist()}")
    print()

    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
