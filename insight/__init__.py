"""PatternKV Insight diagnostics package.

The package is intentionally passive: importing it does not enable observers or
modify model execution. Runtime collection is controlled by environment/config
objects in :mod:`insight.config`.
"""

from insight.config import InsightRuntimeConfig, StandardBaselines, load_standard_baselines
from insight.runtime import abort_sample, begin_sample, end_sample, get_active_observer

__all__ = [
    "InsightRuntimeConfig",
    "StandardBaselines",
    "abort_sample",
    "begin_sample",
    "end_sample",
    "get_active_observer",
    "load_standard_baselines",
]
