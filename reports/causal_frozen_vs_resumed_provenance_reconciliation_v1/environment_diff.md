# Environment Diff

Stored frozen evidence used physical GPU 2 (`GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca`). Stored resumed paper evidence used physical GPU 1 (`GPU-624f86d9-284b-cb46-a671-51d77559dab6`). The canonical A/B in this reconciliation used GPU 1 for both commits and therefore removes physical-GPU mixing as the primary explanation.

The current B1/B4/C4096 slow throughput rows around 275-372 ms/token did not reproduce on GPU 1 in direct current-worker probes. This leaves those stored rows as environment/provenance drift rather than a demonstrated code regression.
