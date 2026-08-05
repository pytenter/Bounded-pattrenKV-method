# Recoverability Decision

Status: `partially_recoverable`

- reason: V mismatch/range_regret exist only in sample records, while K lacks direct range_regret fields and 68 observer files are truncated.
- observer files: `140`
- truncated files: `68`
- dropped sample records: `789376`
- matching_oracle_gap metrics: `conditional_oracle_gap, current_oracle_gap, minmax_oracle_gap`
