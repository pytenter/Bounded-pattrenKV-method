# K-Means Trajectory Analysis

Compiled production helper does not expose per-iteration state. Because identical initial indices still produce divergent final centroids under real delta, trajectory divergence is inferred from final production K-means outputs, not from an instrumented replacement path.
