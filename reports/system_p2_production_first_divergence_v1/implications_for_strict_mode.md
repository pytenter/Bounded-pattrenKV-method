# Implications For Strict Mode

P2 cannot be described as full-transformer batch invariant. Its certified scope is K/V projection-local and Layer0 cache construction for this trace; upstream transformer numerics can still diverge and feed later PatternKV state.
