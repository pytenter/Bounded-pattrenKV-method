# Majority Vote Audit

Maj@8 votes on canonical answer keys, not raw strings.

Flow:

```text
raw generation
-> extract final answer
-> normalize answer
-> canonical key
-> Counter(keys)
-> unique modal key
```

Parser failures contribute no vote. The denominator remains eight responses per problem.

## Verified Fragmentation Case

The following raw variants vote together:

```text
\frac{39}{7}
39/7
\dfrac{39}{7}
\frac{39}{7}
```

With additional votes `5`, `5`, `6`, and one parse failure, the unique modal prediction is:

```text
\frac{39}{7}
```

## Tie Policy

Unchanged:

```text
no unique modal canonical answer -> unresolved -> incorrect
```

Verified 4-vs-4 tie remains incorrect.
