# Post-Graph Bottleneck

CUDA Graph replay is not a valid runtime optimization in this V1 because replayed logits/top1 diverge from eager despite matching coarse cache counters. The prior launch forensic remains the valid bottleneck evidence and points to FP16 tail Value launch fragmentation as the first target.
