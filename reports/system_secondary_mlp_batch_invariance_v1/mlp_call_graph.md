# MLP Call Graph

`LlamaDecoderLayer_PatternKV.forward`: attention output -> residual add -> `post_attention_layernorm` -> `patternkv_mlp_oracle_forward`; inside MLP: gate_proj, up_proj, activation, product, down_proj.
