# Collector Truncation Audit

Status: `core_aggregates_unaffected`

The collector guard truncates only per-sample records. The core aggregate counters used for the published CSV summaries stay intact.
