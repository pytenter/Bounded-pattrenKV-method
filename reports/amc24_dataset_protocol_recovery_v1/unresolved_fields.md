# Unresolved Fields

## Blocking

- exact canonical AMC24 dataset source artifact;
- exact upstream row-selection rule;
- exact problem count;
- stable problem IDs;
- source row IDs;
- ground-truth answer format;
- answer extraction/normalization rule;
- duplicate policy;
- prompt protocol sufficient for paper-comparable AMC;
- sampling protocol sufficient for paper-comparable AMC.

## Nonblocking If Preregistered Later

These fields are not recoverable from PatternKV primary sources, but can be preregistered before any AMC24 GPU run if the dataset identity is solved:

- original seed list;
- project seed list;
- majority tie policy;
- parser ambiguity rule;
- project reuse of AIME24 generation parameters.

## Minimum Information Needed To Unblock

One of the following is required:

1. PatternKV authors release the AMC/AIME Long-CoT evaluation script and data files.
2. PatternKV authors identify the exact NuminaMath revision, split, filter, row IDs, answer extraction, and scoring code.
3. An upstream NuminaMath artifact is found that explicitly defines AMC24 rows and answers in a recoverable way.
4. This project decides to create a new, transparently preregistered AMC24 benchmark independent of PatternKV, with clear labeling that it is not recovered PatternKV AMC24.
