# RMSNorm Boundary Call Graph

`LlamaDecoderLayer_PatternKV.forward`: attention output -> residual add -> `POST_ATTENTION_RMSNORM_INPUT` -> `patternkv_post_attention_rmsnorm` -> `POST_ATTENTION_RMSNORM`.
