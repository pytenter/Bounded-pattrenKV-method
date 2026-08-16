# Page Ownership Contract

Packed/page state belongs to the slot cache that owns the request.

- Active page metadata is rebuilt when active rows are assembled.
- Released slots drop their cache reference and are removed from active mappings.
- No active request may share the same writable slot.
- Reuse requires a fresh cache factory or clean initialized cache for the new request.

The implementation avoids full-pool memset; stale bytes are made unreachable by ownership and valid-length reset.

