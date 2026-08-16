# Lifecycle State Machine

Legal transitions:

- `FREE -> ALLOCATED`
- `ALLOCATED -> ACTIVE`
- `ACTIVE -> FINISHED`
- `FINISHED -> FREE`

Rejected cases:

- activate without allocation
- finish non-active request
- release non-finished request
- release unknown request
- double release
- duplicate request id
- allocate when full
- commit stale row mapping
- decode a released/unknown request

