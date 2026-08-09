# AIME24 Pattern S16 Hadamard Mechanism Diagnostic

## Scope

- Ran only Pattern S16 and Pattern S16 + Hadamard.
- Used frozen 6-task subset from the 12 reference trajectories.
- Used static and pseudo matched FP16 controls at checkpoints 128, 512, 1024, 2048, 4096.
- Did not run VarN, Hadamard+VarN, KVarN full pipeline, accuracy, or full AIME.

## Decisions

- `HADAMARD_STATIC_EFFECT=NONE`
- `HADAMARD_ACCUMULATION_EFFECT=NONE`
- `HADAMARD_NORM_EFFECT=WEAK`
- `NEXT_PRIORITY=VarN isolation / state stabilization`

## Static Effect

- hidden_relative_L2 static degradation AUC: S16 `0.01121939078439027`, S16+Had `0.01121939078439027`, delta `0.0`, improved `0/6`

## Accumulation Effect

- hidden_relative_L2 ACC_AUC: S16 `0.5908144535496831`, S16+Had `0.5958866598084569`, delta `0.005072206258773804`, improved `3/6`
- attention_output_relative_L2 ACC_AUC: S16 `0.5908144535496831`, S16+Had `0.5958866598084569`, delta `0.005072206258773804`, improved `3/6`

## Norm Effect

- k_source_p95: S16 `0.0643781042017508`, S16+Had `0.06390485500996873`, delta `0.0004908680217340783`, improved `3/6`
- k_source_p99: S16 `0.13215649278383293`, S16+Had `0.12413163701086878`, delta `-0.0036284605972463913`, improved `4/6`
- v_source_p95: S16 `0.12063959046572557`, S16+Had `0.12426360872850631`, delta `0.0022548316745087393`, improved `2/6`
- v_source_p99: S16 `0.22979573660733874`, S16+Had `0.23470021875473301`, delta `0.000543754547834438`, improved `3/6`

## Artifact Rows

- Pairwise summary rows: `735`
- Raw CSVs are retained locally; gzipped CSVs are versioned.
