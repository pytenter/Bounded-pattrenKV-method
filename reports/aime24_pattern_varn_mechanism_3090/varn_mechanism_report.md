# AIME24 Pattern S16 VarN-only Mechanism Diagnostic

## Scope

- Ran only Pattern S16 and Pattern S16 + VarN-only.
- Used frozen 6-task subset from the 12 reference trajectories.
- Used static and pseudo matched FP16 controls at checkpoints 128, 512, 1024, 2048, 4096.
- Did not run Hadamard, Hadamard+VarN, KVarN full pipeline, accuracy, or full AIME.

## Decisions

- `VARN_STATIC_EFFECT=NONE`
- `VARN_ACCUMULATION_EFFECT=NONE`
- `VARN_NORM_EFFECT=STRONG`
- `NEXT_PRIORITY=QK / attention-logit / value-direction propagation diagnostic`

## Static Effect

- hidden_relative_L2 static degradation AUC: S16 `0.01121939078439027`, S16+VarN `0.01121939078439027`, delta `0.0`, improved `0/6`

## Accumulation Effect

- hidden_relative_L2 ACC_AUC: S16 `0.5908144535496831`, S16+VarN `0.5703466834092978`, delta `0.0073394570499658585`, improved `2/6`
- attention_output_relative_L2 ACC_AUC: S16 `0.5908144535496831`, S16+VarN `0.5703466834092978`, delta `0.0073394570499658585`, improved `2/6`

## Norm Effect

- k_source_p95: S16 `0.0643781042017508`, S16+VarN `0.06412813600582012`, delta `-0.0029148082016035963`, improved `5/6`
- k_source_p99: S16 `0.13215649278383293`, S16+VarN `0.12599993087831535`, delta `-0.008641356425359889`, improved `5/6`
- v_source_p95: S16 `0.12063959046572557`, S16+VarN `0.11893674483799258`, delta `-0.003398546559037645`, improved `4/6`
- v_source_p99: S16 `0.22979573660733874`, S16+VarN `0.22706596943753538`, delta `-0.011515596709214274`, improved `5/6`

## Artifact Rows

- Pairwise summary rows: `735`
- Raw CSVs are retained locally; gzipped CSVs are versioned.
