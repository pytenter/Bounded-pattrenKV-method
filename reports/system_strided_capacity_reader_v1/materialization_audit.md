# Materialization Audit

| Stream | Materialized by strided candidate | Notes |
| --- | --- | --- |
| V2 payload | NO | Wrapper refuses dtype casts and passes strided view to C++. |
| Scale | NO | Wrapper requires float16 and passes strided view to C++. |
| Zero | NO | Wrapper requires float16 and passes strided view to C++. |
| Mask | NO | Wrapper requires uint8 and passes strided view to C++. |
| Assignment | NO | Wrapper requires uint8/int16/int32 and passes strided view to C++. |

- Historical materialize calls: 0
- Historical materialized bytes: 0
- torch.cat calls: 0
