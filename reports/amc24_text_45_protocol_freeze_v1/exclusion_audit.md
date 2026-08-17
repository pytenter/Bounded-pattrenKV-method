# Exclusion Audit

## Policy

Include problems fully representable as text.

Exclude problems whose required information depends on a non-text figure, diagram, or image absent from the canonical prompt.

No problem is excluded based on difficulty, answer, length, or expected model performance.

## Excluded Problems

| Competition | Problem | Reason |
| --- | ---: | --- |
| AMC12A | 14 | Figure-dependent; excluded by upstream README |
| AMC12A | 18 | Figure-dependent; excluded by upstream README |
| AMC12A | 22 | Figure-dependent; excluded by upstream README |
| AMC12B | 7 | Figure-dependent; excluded by upstream README |
| AMC12B | 19 | Figure-dependent; excluded by upstream README |

## Verification

The canonical dataset contains all remaining problem numbers from 1 through 25 for each competition:

- AMC12A: 1-13, 15-17, 19-21, 23-25
- AMC12B: 1-6, 8-18, 20-25
