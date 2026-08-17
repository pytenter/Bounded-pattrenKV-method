# Answer Parser

## Status

`NOT_IMPLEMENTED_BLOCKED`.

An AMC24 parser was not implemented because the canonical answer space is unresolved.

## Why Parser Implementation Is Blocked

NuminaMath-CoT exposes `solution` text but no separate gold answer field. It does not state whether PatternKV's AMC24 scoring used:

- multiple-choice labels;
- option text;
- numeric final answers extracted from solutions;
- a curated answer file outside NuminaMath;
- a private normalized evaluation subset.

Implementing a parser before resolving this would silently create a benchmark rather than recover one.

## Parser Requirements Once Unblocked

The future parser must be:

- deterministic;
- shared by all methods;
- independent of gold answer during parsing;
- conservative on conflicting candidates;
- covered by targeted tests derived from the true answer format.

If the future answer space is multiple choice, tests should cover boxed labels, explicit final-answer labels, conflicting final answers, invalid labels, empty generations, and truncated generations.
