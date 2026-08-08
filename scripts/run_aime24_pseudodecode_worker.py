from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-like pseudo-decode worker placeholder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path("results/aime24_pseudodecode_3090_8gpu/reference"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/aime24_pseudodecode_3090_8gpu/pseudo"))
    parser.add_argument("--max-checkpoint", type=int, default=24576)
    args = parser.parse_args()
    raise SystemExit(
        json.dumps(
            {
                "status": "not_implemented",
                "reason": "Pseudo worker must be completed against production cache internals before formal run approval.",
                "config": args.config,
                "gpu_id": args.gpu_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
