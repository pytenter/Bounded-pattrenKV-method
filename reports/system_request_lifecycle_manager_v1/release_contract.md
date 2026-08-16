# Release Contract

Release is allowed only after `FINISHED`.

Release actions:

- remove request id from `request_to_slot`
- remove request id from active mapping
- free centroid state if present
- drop slot cache reference
- return slot to free-list exactly once

Double release and unknown release are rejected.

