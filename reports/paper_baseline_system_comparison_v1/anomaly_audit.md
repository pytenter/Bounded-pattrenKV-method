# Anomaly Audit

CAUSAL C2048 B2 repeats were [412.6601162715815, 288.94841199507937, 283.6537666444201] ms/token; the median is 288.948 ms/token. All repeats preserved decode-only timing, true batch, zero serial dispatch, and zero fallback. The elevated repeat is retained in raw evidence and the reported primary statistic is the median across all repeats. The resumed median differs from the frozen historical B2 reference (216.8267 ms/token); this report does not overwrite the frozen result or attribute the difference to a new optimization.
