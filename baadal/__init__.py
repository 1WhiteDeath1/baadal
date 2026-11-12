"""Baadal: forecast -> flood map -> budget-aware flood-prevention planning with a QAOA step.

Baadal consumes rainfall / river-inflow forecasts. It does not predict weather.
The flood and damage models are classical. The only quantum component is the
intervention-selection optimisation (QUBO solved with QAOA in Qiskit).
All damage, cost and population numbers in the bundled presets are illustrative.
"""

__version__ = "0.1.0"
