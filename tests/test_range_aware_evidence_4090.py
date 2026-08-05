from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_existing_recoverability_audit_artifacts_present():
    report_root = ROOT / "reports" / "insight_v2" / "range_aware_evidence_4090"
    assert (report_root / "recoverability_decision.json").exists()
    payload = json.loads((report_root / "recoverability_decision.json").read_text(encoding="utf-8"))
    assert payload.get("status") in {"partially_recoverable", "supported", "not_supported"}
