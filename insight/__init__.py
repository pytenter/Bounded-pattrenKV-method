"""PatternKV Insight diagnostics package.

The package is intentionally passive: importing it does not enable observers or
modify model execution. Runtime collection is controlled by environment/config
objects in :mod:`insight.config`.
"""

from insight.config import InsightRuntimeConfig, StandardBaselines, load_standard_baselines

__all__ = ["InsightRuntimeConfig", "StandardBaselines", "load_standard_baselines"]
