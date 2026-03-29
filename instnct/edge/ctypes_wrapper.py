"""
ctypes wrapper for C forward pass
===================================
Replaces Python rollout_token with C version for ~29x speedup.
"""
import ctypes, os, numpy as np
from numpy.ctypeslib import ndpointer

# Load DLL
_dll_path = os.path.join(os.path.dirname(__file__), 'instnct_edge.dll')
if not os.path.exists(_dll_path):
    raise FileNotFoundError(f"Compile first: gcc -O2 -shared -o instnct_edge.dll instnct_edge.c instnct_lut.c")

_lib = ctypes.cdll.LoadLibrary(_dll_path)

# Struct definitions matching C
class InstnctNeuron(ctypes.Structure):
    _fields_ = [("theta", ctypes.c_uint8),
                 ("channel", ctypes.c_uint8),
                 ("polarity", ctypes.c_uint8),
                 ("refr_period", ctypes.c_uint8)]

class InstnctEdge(ctypes.Structure):
    _fields_ = [("source", ctypes.c_uint16),
                 ("target", ctypes.c_uint16)]

class InstnctState(ctypes.Structure):
    _fields_ = [("charge", ctypes.c_int8 * 256),
                 ("refr_counter", ctypes.c_uint8 * 256)]

# Function signatures
_lib.instnct_rollout_token.restype = None
_lib.instnct_rollout_token.argtypes = [
    ctypes.POINTER(InstnctNeuron),  # neurons
    ctypes.POINTER(InstnctEdge),     # edges
    ctypes.c_int,                    # n_edges
    ctypes.POINTER(ctypes.c_uint16), # sdr_active
    ctypes.c_int,                    # n_active
    ctypes.POINTER(InstnctState),    # state
    ctypes.POINTER(ctypes.c_int8),   # spike_out
]

_lib.instnct_state_reset.restype = None
_lib.instnct_state_reset.argtypes = [ctypes.POINTER(InstnctState)]

_lib.instnct_readout_simple.restype = ctypes.c_uint8
_lib.instnct_readout_simple.argtypes = [ctypes.POINTER(InstnctState)]

def build_c_network(mask, theta, channel, polarity_bool, refr_period=None):
    """Convert numpy arrays to C structs."""
    H = mask.shape[0]
    rows, cols = np.where(mask)
    n_edges = len(rows)

    # Neurons
    neurons = (InstnctNeuron * H)()
    for i in range(H):
        neurons[i].theta = int(np.clip(theta[i], 1, 15))
        neurons[i].channel = int(np.clip(channel[i], 1, 8))
        neurons[i].polarity = 1 if polarity_bool[i] else 0
        neurons[i].refr_period = int(refr_period[i]) if refr_period is not None else 1

    # Edges
    edges = (InstnctEdge * n_edges)()
    for e in range(n_edges):
        edges[e].source = int(rows[e])
        edges[e].target = int(cols[e])

    return neurons, edges, n_edges

def c_rollout_token(neurons, edges, n_edges, sdr_indices, state):
    """Run one token through C forward pass."""
    H = 256
    n_active = len(sdr_indices)
    sdr_arr = (ctypes.c_uint16 * n_active)(*sdr_indices.astype(np.uint16))
    spike_out = (ctypes.c_int8 * H)()

    _lib.instnct_rollout_token(neurons, edges, n_edges, sdr_arr, n_active,
                                ctypes.byref(state), spike_out)

    # Convert spike and charge to numpy
    charge = np.array(state.charge[:], dtype=np.int8)
    spike = np.array(spike_out[:], dtype=np.int8)
    return spike, charge

def new_state():
    """Create fresh state."""
    state = InstnctState()
    _lib.instnct_state_reset(ctypes.byref(state))
    return state

if __name__ == "__main__":
    # Quick test
    import time
    H = 256
    mask = (np.random.RandomState(42).rand(H,H) < 0.05).astype(bool)
    np.fill_diagonal(mask, False)
    theta = np.full(H, 6, np.uint8)
    channel = np.random.RandomState(42).randint(1, 9, size=H).astype(np.uint8)
    polarity = np.random.RandomState(42).rand(H) > 0.1

    neurons, edges, n_edges = build_c_network(mask, theta, channel, polarity)
    state = new_state()

    # SDR for byte 65
    rng = np.random.RandomState(42)
    sdr = rng.choice(158, size=32, replace=False).astype(np.uint16)

    t0 = time.perf_counter()
    for _ in range(10000):
        spike, charge = c_rollout_token(neurons, edges, n_edges, sdr, state)
    t1 = time.perf_counter()
    print(f"C ctypes: {10000/(t1-t0):.0f} tokens/sec")
    print(f"Spike sum: {spike.sum()}, Charge mean: {charge.mean():.2f}")
