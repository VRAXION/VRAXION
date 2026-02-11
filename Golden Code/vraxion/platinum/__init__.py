"""Platinum Code — Production-ready VRAXION runtime.

Clean extraction from Golden Code (vraxion.instnct).
One module = one concern. No AGC, no dead code.

Public API
----------
Models:
    AbsoluteHallway, Prismion, PrismionState

Configuration:
    Settings, load_settings

Control Loops:
    BrainstemMixer, BrainstemMixerConfig,
    ThermostatParams, AGCParams, InertiaAutoParams,
    CadenceGovernor, PanicReflex,
    apply_thermostat, apply_update_agc, apply_inertia_auto

Routing:
    LocationExpertRouter

Swarm & Budget:
    fibonacci_halving_budget

Checkpoints:
    save_modular_checkpoint, load_modular_checkpoint,
    resolve_modular_resume_dir
"""

__version__ = "0.1.0"

# Core Neural Network Models
from .hallway import AbsoluteHallway
from .swarm import Prismion, PrismionState

# Configuration & Settings
from .settings import Settings, load_settings

# Control Loop Components
from .brainstem import BrainstemMixer, BrainstemMixerConfig
from vraxion.instnct.controls import (
    ThermostatParams,
    AGCParams,
    InertiaAutoParams,
    CadenceGovernor,
    PanicReflex,
    apply_thermostat,
    apply_update_agc,
    apply_inertia_auto,
)

# Expert Routing
from .experts import LocationExpertRouter

# Swarm & Budget Allocation
from .swarm import fibonacci_halving_budget

# Checkpoint I/O
from .checkpoint import (
    save_modular_checkpoint,
    load_modular_checkpoint,
    resolve_modular_resume_dir,
)

__all__ = [
    # Models
    "AbsoluteHallway",
    "Prismion",
    "PrismionState",
    # Configuration
    "Settings",
    "load_settings",
    # Control Loops
    "BrainstemMixer",
    "BrainstemMixerConfig",
    "ThermostatParams",
    "AGCParams",
    "InertiaAutoParams",
    "CadenceGovernor",
    "PanicReflex",
    "apply_thermostat",
    "apply_update_agc",
    "apply_inertia_auto",
    # Routing & Experts
    "LocationExpertRouter",
    # Swarm
    "fibonacci_halving_budget",
    # Checkpoints
    "save_modular_checkpoint",
    "load_modular_checkpoint",
    "resolve_modular_resume_dir",
]
