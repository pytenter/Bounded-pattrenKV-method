# Remote Branch Cleanup Candidates

No branch deletion was performed. Every entry below requires manual approval before any destructive action.

## `bounded/analysis/patternkv-vgate-layer-head-opportunity`

- HEAD SHA: `995034340dc48c3068f141ca157a9f1c37c5b5df`
- Last commit: 2026-08-05 22:30:28 +0800 `analysis: localize cross-hardware V gate false negatives`
- Merged into system: False
- Unique commits vs system: system_only=115;branch_only=2;merged_into_system=False
- Candidate assessment: has unique commits vs system; recommended action `REVIEW_REQUIRED`.
- Recovery command before deletion: `git branch <local-name> 995034340dc48c3068f141ca157a9f1c37c5b5df` or `git fetch bounded 995034340dc48c3068f141ca157a9f1c37c5b5df`

## `bounded/exp/aime-int2-wave1-v100-8gpu`

- HEAD SHA: `20caa33d93c470ae1382a63300de509197892c29`
- Last commit: 2026-08-10 23:04:07 +0800 `docs: report AIME24 full30 Sink validation`
- Merged into system: False
- Unique commits vs system: system_only=80;branch_only=2;merged_into_system=False
- Candidate assessment: has unique commits vs system; recommended action `REVIEW_REQUIRED`.
- Recovery command before deletion: `git branch <local-name> 20caa33d93c470ae1382a63300de509197892c29` or `git fetch bounded 20caa33d93c470ae1382a63300de509197892c29`

## `bounded/exp/aime-pattern-hadamard-mechanism-3090`

- HEAD SHA: `2d9928fcbd5dfd2dfecb5f2cf1b70a641983f3eb`
- Last commit: 2026-08-09 16:21:21 +0800 `run: complete Hadamard accumulation diagnostic`
- Merged into system: False
- Unique commits vs system: system_only=73;branch_only=2;merged_into_system=False
- Candidate assessment: has unique commits vs system; recommended action `REVIEW_REQUIRED`.
- Recovery command before deletion: `git branch <local-name> 2d9928fcbd5dfd2dfecb5f2cf1b70a641983f3eb` or `git fetch bounded 2d9928fcbd5dfd2dfecb5f2cf1b70a641983f3eb`

## `bounded/exp/aime-pattern-varn-mechanism-3090`

- HEAD SHA: `2f63bddef151df0f32a40d51c73f125a8089800e`
- Last commit: 2026-08-09 18:15:48 +0800 `run: complete Pattern S16 VarN diagnostic`
- Merged into system: False
- Unique commits vs system: system_only=72;branch_only=2;merged_into_system=False
- Candidate assessment: has unique commits vs system; recommended action `REVIEW_REQUIRED`.
- Recovery command before deletion: `git branch <local-name> 2f63bddef151df0f32a40d51c73f125a8089800e` or `git fetch bounded 2f63bddef151df0f32a40d51c73f125a8089800e`

## `bounded/exp/aime-pseudodecode-3090-8gpu`

- HEAD SHA: `f7f6ca9954daa76cb702941f1b018ae294c0e378`
- Last commit: 2026-08-09 15:26:12 +0800 `audit: isolate canonical VarN semantics`
- Merged into system: True
- Unique commits vs system: system_only=72;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> f7f6ca9954daa76cb702941f1b018ae294c0e378` or `git fetch bounded f7f6ca9954daa76cb702941f1b018ae294c0e378`

## `bounded/exp/aime-qk-routing-vdirection-3090`

- HEAD SHA: `0a644856f2a45569e489b91ae452253560484e23`
- Last commit: 2026-08-09 19:03:25 +0800 `run: complete routing-value propagation diagnostic`
- Merged into system: True
- Unique commits vs system: system_only=68;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 0a644856f2a45569e489b91ae452253560484e23` or `git fetch bounded 0a644856f2a45569e489b91ae452253560484e23`

## `bounded/exp/aime-selective-value-precision-3090`

- HEAD SHA: `241b832aee31c0e328d13675efb7819508c29ac9`
- Last commit: 2026-08-10 09:40:44 +0800 `run: complete selective Value precision screen`
- Merged into system: True
- Unique commits vs system: system_only=62;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 241b832aee31c0e328d13675efb7819508c29ac9` or `git fetch bounded 241b832aee31c0e328d13675efb7819508c29ac9`

## `bounded/exp/aime-value-capacity-budget-3090`

- HEAD SHA: `83c46ed1252a32ca42dcb81e172bd3e4c0a060a0`
- Last commit: 2026-08-10 11:59:03 +0800 `run: complete Value capacity budget study`
- Merged into system: True
- Unique commits vs system: system_only=61;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 83c46ed1252a32ca42dcb81e172bd3e4c0a060a0` or `git fetch bounded 83c46ed1252a32ca42dcb81e172bd3e4c0a060a0`

## `bounded/exp/aime-value-direction-screen-3090`

- HEAD SHA: `47cf601fcfd4b20fa3823fe540b1d48ca9920d7d`
- Last commit: 2026-08-09 20:14:55 +0800 `run: complete direction-aware Value mechanism screen`
- Merged into system: True
- Unique commits vs system: system_only=65;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 47cf601fcfd4b20fa3823fe540b1d48ca9920d7d` or `git fetch bounded 47cf601fcfd4b20fa3823fe540b1d48ca9920d7d`

## `bounded/exp/aime-value-objective-screen-3090`

- HEAD SHA: `fd6748b9f93c8b357c5643e07d8e482438bdcd45`
- Last commit: 2026-08-09 19:21:41 +0800 `exp: audit mechanism-guided Value objectives`
- Merged into system: True
- Unique commits vs system: system_only=67;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> fd6748b9f93c8b357c5643e07d8e482438bdcd45` or `git fetch bounded fd6748b9f93c8b357c5643e07d8e482438bdcd45`

## `bounded/exp/patternkv-4090-range-aware-targeted`

- HEAD SHA: `03db01ee96ec7a46da5baa0dcc6395eefa410370`
- Last commit: 2026-08-06 00:12:53 +0800 `feat: prepare 4090 targeted range-aware collection`
- Merged into system: False
- Unique commits vs system: system_only=115;branch_only=5;merged_into_system=False
- Candidate assessment: has unique commits vs system; recommended action `REVIEW_REQUIRED`.
- Recovery command before deletion: `git branch <local-name> 03db01ee96ec7a46da5baa0dcc6395eefa410370` or `git fetch bounded 03db01ee96ec7a46da5baa0dcc6395eefa410370`

## `bounded/exp/patternkv-insight-wave-a-4090-runtime6c88`

- HEAD SHA: `6d6cdcb74d0e4611a9274660dfdc132161b5318b`
- Last commit: 2026-08-05 17:41:31 +0800 `fix: preserve single-4090 sample selection`
- Merged into system: False
- Unique commits vs system: system_only=117;branch_only=3;merged_into_system=False
- Candidate assessment: has unique commits vs system; recommended action `REVIEW_REQUIRED`.
- Recovery command before deletion: `git branch <local-name> 6d6cdcb74d0e4611a9274660dfdc132161b5318b` or `git fetch bounded 6d6cdcb74d0e4611a9274660dfdc132161b5318b`

## `bounded/exp/patternkv-insight-wave-a-4gpu`

- HEAD SHA: `ce4ef0749680aebe016aeaa06e6e22ff9d711167`
- Last commit: 2026-08-06 15:24:10 +0800 `docs: publish completed AIME24 results`
- Merged into system: True
- Unique commits vs system: system_only=114;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> ce4ef0749680aebe016aeaa06e6e22ff9d711167` or `git fetch bounded ce4ef0749680aebe016aeaa06e6e22ff9d711167`

## `bounded/exp/patternkv-longbench-data-parity-wave-a`

- HEAD SHA: `0cc1a8dc838ae5c9365fea701cfcd65a9c0c0b63`
- Last commit: 2026-08-04 23:06:17 +0800 `Restore LongBench parity and Insight Wave A gate`
- Merged into system: True
- Unique commits vs system: system_only=119;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 0cc1a8dc838ae5c9365fea701cfcd65a9c0c0b63` or `git fetch bounded 0cc1a8dc838ae5c9365fea701cfcd65a9c0c0b63`

## `bounded/exp/patternkv-parity-microsmoke-wave-a`

- HEAD SHA: `f1e074b8272aa80d062d6cd852ab1fe036ab2afb`
- Last commit: 2026-08-04 22:08:28 +0800 `exp: audit Insight parity gate execution`
- Merged into system: True
- Unique commits vs system: system_only=120;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> f1e074b8272aa80d062d6cd852ab1fe036ab2afb` or `git fetch bounded f1e074b8272aa80d062d6cd852ab1fe036ab2afb`

## `bounded/insight/patternkv-diagnostics-v1`

- HEAD SHA: `9d0afb7ef470253cb16fdb02c34d4685e44c65d2`
- Last commit: 2026-08-04 20:32:46 +0800 `feat: add PatternKV insight diagnostics and oracle analysis`
- Merged into system: True
- Unique commits vs system: system_only=123;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 9d0afb7ef470253cb16fdb02c34d4685e44c65d2` or `git fetch bounded 9d0afb7ef470253cb16fdb02c34d4685e44c65d2`

## `bounded/insight/patternkv-observer-wave-a`

- HEAD SHA: `4b877cdf471bc749526570f325f1e1fca3666329`
- Last commit: 2026-08-04 20:59:10 +0800 `feat: connect PatternKV observer and run Wave A diagnostics`
- Merged into system: True
- Unique commits vs system: system_only=122;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 4b877cdf471bc749526570f325f1e1fca3666329` or `git fetch bounded 4b877cdf471bc749526570f325f1e1fca3666329`

## `bounded/insight/patternkv-runner-parity-wave-a`

- HEAD SHA: `18e2f788ba47225e251a94cb2606d53e2203294a`
- Last commit: 2026-08-04 21:34:07 +0800 `feat: connect real insight runner and validate Wave A parity`
- Merged into system: True
- Unique commits vs system: system_only=121;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 18e2f788ba47225e251a94cb2606d53e2203294a` or `git fetch bounded 18e2f788ba47225e251a94cb2606d53e2203294a`

## `bounded/repro/patternkv-longbench-8k-single4090`

- HEAD SHA: `2fc966681edca1a932c61ac77d956bbbda998833`
- Last commit: 2026-08-04 17:38:48 +0800 `Add 4090 LongBench 21x50 8K results`
- Merged into system: True
- Unique commits vs system: system_only=127;branch_only=0;merged_into_system=True
- Candidate assessment: merged into system or no unique commits vs system; recommended action `ARCHIVE_CANDIDATE`.
- Recovery command before deletion: `git branch <local-name> 2fc966681edca1a932c61ac77d956bbbda998833` or `git fetch bounded 2fc966681edca1a932c61ac77d956bbbda998833`
