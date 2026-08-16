# Allocation Reset Contract

Allocation assigns a free persistent slot to a request id.

Required initialized state:

- request total token length
- packed K/V/V4 lengths
- recent and pending lengths/content
- centroid state and active bit
- causal importance state
- precision mask and selector-derived metadata
- page ownership metadata if present

The implementation uses a cache factory at allocation/reuse time. A reused slot receives a new `PatternQuantizedKVCache`; old semantic state is not reused.

