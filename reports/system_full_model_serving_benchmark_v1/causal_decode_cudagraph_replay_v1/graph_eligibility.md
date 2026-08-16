# Graph Eligibility

V1 eligibility:

- prefill complete
- decode-only
- fixed active batch size
- no membership change
- no refill
- fixed decode horizon captured before timed replay
- fixed per-step cache shapes available

Unsupported situations fall back to eager: membership changes, dynamic add/remove, ragged shape changes not represented by a captured graph sequence, and page-boundary transitions outside the captured step horizon.
