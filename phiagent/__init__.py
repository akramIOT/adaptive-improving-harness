"""Compatibility import for the renamed :mod:`adaptive_harness` package."""

import sys

import adaptive_harness
from adaptive_harness._compat import enable_legacy_imports

enable_legacy_imports()
sys.modules[__name__] = adaptive_harness
