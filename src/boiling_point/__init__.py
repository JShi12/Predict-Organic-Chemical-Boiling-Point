# torch, scikit-learn and xgboost each ship their own OpenMP runtime. On
# macOS, if torch's is loaded first, xgboost segfaults on fit; loading
# xgboost first is safe. Importing it here gives any `boiling_point` import
# the safe order. (active_learning.py also keeps torch single-threaded,
# which is needed too.)
import sys
import warnings

if sys.platform == "darwin" and "torch" in sys.modules and "xgboost" not in sys.modules:
    warnings.warn(
        "torch was imported before boiling_point/xgboost; on macOS this can make "
        "xgboost segfault. Import boiling_point (or xgboost) before torch.",
        RuntimeWarning,
    )

import xgboost  # noqa: E402,F401
