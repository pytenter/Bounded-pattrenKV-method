# Before Fix Mapping

Before S6-B.3.4K, `update_value_causal_importance` computed `mass = attn_weights.mean(dim=1).sum(dim=1)` and executed `v_causal_importance[:, :width] += mass[:, :width]`. That treated physical attention positions as logical importance destinations. S6-B.3.4J proved B1 matched golden while ragged did not, and peer length changed request A's production importance.
