# Reader Layout Examples

- Logical tokens: 8192
- Capacity tokens: 32768
- V2 payload storage stride: (2097152, 262144, 8, 1)
- V2 payload logical view stride: (2097152, 262144, 8, 1)
- Scale storage stride: (262144, 32768, 1, 1)
- Scale logical view stride: (262144, 32768, 1, 1)
- Zero storage stride: (262144, 32768, 1, 1)
- Mask storage stride: (262144, 32768, 1)
- Assignment storage stride: (262144, 32768, 1)
- Unused V2 payload bytes: 6291456
- Slack values are sentinel-filled and correctness checks fail if they are read into output.
