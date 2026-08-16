# Protocol

Each formal point runs in an independent Python subprocess. Decode timing starts after prefill, CUDA synchronization, allocator warmup, and decode-window counter reset. Standard decode=8 points require zero prefill, refill, membership changes, and page-pack calls in the timed window. Long-decode points allow boundary page-pack work and report it separately.
