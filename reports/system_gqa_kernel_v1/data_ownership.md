# Data Ownership

| Data | Q-head specific | KV-head shared | Can reuse? |
|---|---:|---:|---|
| attention alpha | YES | NO | NO |
| packed V2 | NO | YES | YES |
| packed V4 | NO | YES | YES, not implemented in S2B-3 |
| scale | NO | YES | YES |
| zero | NO | YES | YES |
| Pattern mask | NO | YES | YES |
| assignment | NO | YES | YES |
| centroid table | NO | YES | YES |
| recent V | NO | YES | YES, but candidate A did not stage it |
| histogram SAcc | YES | NO | NO |
| final output | YES | NO | NO |

Attention alpha, SAcc, and output remain private per Q head in the candidate.
