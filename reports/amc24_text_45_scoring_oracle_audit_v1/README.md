# AMC24-Text-45 Scoring Oracle Audit V1

Status: `AMC24_TEXT_45_SCORING_ORACLE_AUDIT_V1_SUPPORTED`.

This audit freezes deterministic representation normalization for AMC24-Text open-answer scoring before any AMC24-Text model result is generated.

Scope:

- answer extraction and normalization only;
- no dataset content change;
- no prompt/sampling/seed/protocol generation change;
- no model load;
- no GPU generation.

Primary outputs:

- `answer_form_inventory.json`
- `answer_form_inventory.md`
- `canonical_answer_inventory.json`
- `normalization_rules.md`
- `equivalence_tests.md`
- `non_equivalence_tests.md`
- `parser_audit.md`
- `majority_vote_audit.md`
- `collision_audit.md`
- `final_gate.json`

The scorer canonicalizes bounded syntactic variants for the answer forms actually present in the 45 frozen rows. It does not use gold-aware matching, an LLM judge, decimal tolerance, or unrestricted symbolic equivalence.
