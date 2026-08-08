from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh-state static checkpoint worker placeholder.")
    parser.add_argument("--queue", type=Path, default=Path("run/aime24_pseudodecode_3090_8gpu/static_jobs.jsonl"))
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/aime24_pseudodecode_3090_8gpu/static"))
    args = parser.parse_args()
    raise SystemExit(
        json.dumps(
            {
                "status": "not_implemented",
                "reason": "Static worker must rebuild each checkpoint from a clean FP16 prefix and discard cache state.",
                "queue": str(args.queue),
                "gpu_id": args.gpu_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
