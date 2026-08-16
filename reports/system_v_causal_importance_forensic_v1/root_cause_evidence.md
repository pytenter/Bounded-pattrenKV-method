# Root Cause Evidence

- classification: `V_CAUSAL_IMPORTANCE_METADATA_SEMANTICS_DIVERGENCE`
- previous importance exact: `True`
- raw Q exact: `True`
- Q RoPE exact: `True`
- canonical attention probabilities exact: `True`
- canonical importance mass exact: `True`
- physical mapping exact: `False`
- B1 production matches canonical golden: `True`
- ragged production matches canonical golden: `True`

The first non-equivalent item is not Q projection or attention probability semantics. It is the index mapping used when the physical ragged attention vector is accumulated into a logical per-request importance vector.
