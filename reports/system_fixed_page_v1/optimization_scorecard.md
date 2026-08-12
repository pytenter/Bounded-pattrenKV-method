# Optimization Scorecard

| Metric | Contiguous | Fixed Page | Change |
|---|---:|---:|---:|
| 32K old bytes copied/token | 604063.875 | 4096.000 | 99.32% reduction |
| 32K torch.cat events/token | 2.047 | 0.000 | eliminated in storage ABI |
| 32K mutation latency/token | 91.221 us | 98.489 us | 0.926x |
| 16K mutation latency/token | 90.465 us | 98.890 us | 0.915x |
| page allocations / 128 decode @32K | N/A | 1536 | lazy pages |
| peak allocated memory @32K | 19165696 | 21313024 | +11.20% |
| peak reserved memory @32K | 25165824 | 35651584 | +41.67% |
| TPOT 16K | NOT_RUN | NOT_RUN | page-native reader absent |
| TPOT 32K | NOT_RUN | NOT_RUN | page-native reader absent |
| correctness | PASS | PASS | equivalent storage |

Latency is not faster in this Python storage benchmark; the win is storage ABI/copy elimination, not production E2E yet.
