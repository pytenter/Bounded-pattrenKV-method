#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def main() -> None:
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    local_dir = Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading via ModelScope: {repo_id}", flush=True)
    print(f"local_dir={local_dir}", flush=True)
    path = snapshot_download(repo_id, local_dir=str(local_dir))
    print(f"downloaded_path={path}", flush=True)


if __name__ == "__main__":
    main()
