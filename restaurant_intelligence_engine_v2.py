"""
Compatibility wrapper module so you can `import restaurant_intelligence_engine_v2 as rie`.
This simply exposes the main API from the existing `resturantv1.py` file.
"""

# Import the implementation module (same folder)
import resturantv1 as _impl

# Re-expose the commonly used symbols
CONFIG = _impl.CONFIG
run_full_analysis_v2 = _impl.run_full_analysis_v2

# Convenience: expose the whole implementation module as an attribute
resturantv1 = _impl

__all__ = ["CONFIG", "run_full_analysis_v2", "resturantv1"]
