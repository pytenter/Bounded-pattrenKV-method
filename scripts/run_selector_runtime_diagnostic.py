#!/usr/bin/env python3
"""Prepared matched runtime diagnostic launcher.

The default mode only emits the pre-registered diagnostic protocol. Use
`--print-commands` after formal completion to see explicit commands; this script
does not run CUDA work unless a future operator adds a run mode deliberately.
"""

from __future__ import annotations

import argparse
import json


IDENTITIES = [{"problem_id": p, "base_seed": 42, "sample_id": 0} for p in [0, 5, 10, 15, 20]]
METHODS = ["importance_only25", "error_only25", "CAUSAL_V4_25"]


def protocol() -> dict:
    return {
        "matched_runtime_diagnostic_prepared": True,
        "matched_runtime_diagnostic_run": False,
        "methods": METHODS,
        "identities": IDENTITIES,
        "hardware": "single isolated RTX3090, one process at a time",
        "tier1": {"max_new_tokens": 2048, "purpose": "short path diagnostic"},
        "tier2": {"max_new_tokens": 32768, "purpose": "one formal-cap spot check if needed"},
        "timing": "torch.cuda.synchronize(); time.perf_counter(); model.generate(...); torch.cuda.synchronize()",
        "claim_scope": "diagnostic only, never part of formal quality denominator",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    payload = protocol()
    if args.print_commands:
        payload["commands"] = [
            "# Run only after current formal completion or on an idle RTX3090.",
            "# Use a future measured runner with instrumentation enabled; do not reuse completion timestamps.",
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
