Archived CPU sweeps that still depended on the legacy signal/grow/intensity/mood
controller surface.

These files are preserved as historical references only. They are not part of
the active current-canon CPU sweep tree, which now centers on:

- charge_rate_sweep.py
- dynamic_threshold_sweep.py
- leak_discrete_sweep.py

If one of these archived files needs to come back, port it through
tests/harness/cpu_parameter_sweeps.py instead of reviving the old controller
surface directly.
