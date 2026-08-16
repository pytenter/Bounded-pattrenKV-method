# Read/Write Identity Audit

Recent K write/update/read uses the active batch row inside the currently materialized cache tensor. For B1 A and ragged A in `[A,B]`, active row is 0 in both. Reorder oracle covers active row 1.
