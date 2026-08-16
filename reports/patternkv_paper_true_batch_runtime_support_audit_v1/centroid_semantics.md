# Centroid Semantics

The leading dimension in `[2,8,48,128]` is the active batch row for two logical requests. Each row maps to a request-local centroid slot through `centroid_state_indices`.

- `B`: active batch row
- `8`: KV heads
- `48`: allocated centroid bank width; valid count is request-local
- `128`: head dimension

Assignments are also request-local: `v_assignment_idx[B,Hkv,T]`. A shared row-0 centroid bank is semantically invalid and is covered by a negative-control regression test.
