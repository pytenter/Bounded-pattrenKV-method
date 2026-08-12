# Compact Stream Mapping

- V2 payload pages use extension-native `[B*H_kv, D/16, page_size]` layout.
- Scale/zero pages use `[B*H_kv, D/group_size, page_size]`.
- Mask and assignment pages use `[B, H_kv, page_size]`.
