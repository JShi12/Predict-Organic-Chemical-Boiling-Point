# Load xgboost's OpenMP runtime before any test module imports torch
# (see src/boiling_point/__init__.py).
import boiling_point  # noqa: F401
