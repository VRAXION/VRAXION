#!/usr/bin/env python
"""Test script to verify platinum API exports work correctly."""

import sys
from pathlib import Path

# Add Golden Code to path
golden_code_path = Path(__file__).parent / "Golden Code"
sys.path.insert(0, str(golden_code_path))

print("Testing platinum API exports...")
print()

# Test 1: Import all public API items
try:
    from vraxion.platinum import (
        # Models
        AbsoluteHallway,
        Prismion,
        PrismionState,
        # Configuration
        Settings,
        load_settings,
        # Control Loops
        BrainstemMixer,
        BrainstemMixerConfig,
        ThermostatParams,
        AGCParams,
        InertiaAutoParams,
        CadenceGovernor,
        PanicReflex,
        apply_thermostat,
        apply_update_agc,
        apply_inertia_auto,
        # Routing
        LocationExpertRouter,
        # Swarm
        fibonacci_halving_budget,
        # Checkpoints
        save_modular_checkpoint,
        load_modular_checkpoint,
        resolve_modular_resume_dir,
    )
    print("✓ All imports from vraxion.platinum successful")
except ImportError as e:
    print(f"✗ Import from vraxion.platinum failed: {e}")
    sys.exit(1)

# Test 2: Verify __all__ is defined
try:
    import vraxion.platinum
    assert hasattr(vraxion.platinum, "__all__"), "__all__ not defined"
    print(f"✓ __all__ is defined with {len(vraxion.platinum.__all__)} exports")
except AssertionError as e:
    print(f"✗ {e}")
    sys.exit(1)

# Test 3: Verify backward compatibility (old import paths still work)
try:
    from vraxion.platinum.hallway import AbsoluteHallway as AH2
    from vraxion.platinum.brainstem import BrainstemMixer as BM2
    from vraxion.platinum.settings import Settings as S2
    print("✓ Backward compatible imports still work")
except ImportError as e:
    print(f"✗ Backward compatible import failed: {e}")
    sys.exit(1)

# Test 4: Verify objects are the same (not duplicated)
try:
    assert AbsoluteHallway is AH2, "AbsoluteHallway object differs"
    assert BrainstemMixer is BM2, "BrainstemMixer object differs"
    assert Settings is S2, "Settings object differs"
    print("✓ New and old imports reference same objects")
except AssertionError as e:
    print(f"✗ {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✓ All tests passed! Platinum API exports working correctly.")
print("=" * 60)
