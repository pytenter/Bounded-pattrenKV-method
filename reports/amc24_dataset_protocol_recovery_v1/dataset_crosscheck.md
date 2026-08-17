# Dataset Crosscheck

No canonical AMC24 dataset was generated, so dataset content checks are `NOT_RUN`.

## Checks Blocked

| Check | Status | Reason |
| --- | --- | --- |
| row count check | NOT_RUN | AMC24 row-selection rule unresolved |
| unique problem_id check | NOT_RUN | no canonical problem IDs |
| duplicate problem text check | NOT_RUN | no canonical rows |
| missing answer check | NOT_RUN | no ground-truth field |
| invalid answer check | NOT_RUN | answer space unresolved |
| dataset SHA256 | NOT_RUN | no dataset file |
| deterministic regeneration | NOT_RUN | no source filter to regenerate |

## Minimum Required To Run Checks

A valid crosscheck requires a primary-source or project-approved frozen rule with:

- upstream dataset/repository revision;
- split/config;
- exact filter expression;
- expected problem count;
- answer extraction/normalization rule;
- duplicate policy;
- stable source row identifiers.
