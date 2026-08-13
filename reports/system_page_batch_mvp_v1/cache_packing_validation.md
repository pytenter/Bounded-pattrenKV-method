# Cache Packing Validation

- Packing uses request-local precision rows and never treats `precision_mask[0]` as a batch-global layout.
- B=2/B=4 cases include different masks and different page-local V2/V4 counts.
- `page_mapping_validation.json`, `scale_zero_alignment.json`, and `pattern_metadata_alignment.json` contain replayable gate outputs.
