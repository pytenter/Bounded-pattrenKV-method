# Root Cause Analysis

`UPSTREAM_TRANSFORMER_BATCH_NUMERICAL_DRIFT`. BI K/V projection and Layer0 PatternKV cache construction are exact; the first real non-exact tensor is Layer0 `MLP_OUTPUT`, an ordinary transformer block numerical drift.
