"""temporal-leakage-audit: a temporal leakage-response instrument for clinical trial outcome prediction.

Modules: ``features`` (as-of-time-censored feature matrices), ``leakage`` (leakage-response
curve and LAP), ``causal`` (positivity-aware causal layer), ``decision`` (policy evaluation),
``splits``, ``metrics``, ``models``, ``config``, ``report``; ``data.connectors`` builds the
drug-program benchmark from public APIs and ``data.synth`` generates the synthetic testbed.
"""
__version__ = "1.0.0"
