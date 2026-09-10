import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_contact_audit import combine  # noqa: E402


def test_missing_review_cannot_be_reported_as_complete():
    with pytest.raises(ValueError, match="incomplete"):
        combine(pd.DataFrame({"review_id": [0, 1]}), [pd.DataFrame({"review_id": [0]})])


def test_contradictory_timing_is_flagged_without_silent_correction():
    review = pd.DataFrame({"review_id": [0], "contact_lower_ms": [200], "contact_upper_ms": [400],
                           "t0_status": ["post"], "body_part": ["foot"], "confidence": ["medium"],
                           "note": ["Ball leaves foot in later frames"]})
    result = combine(pd.DataFrame({"review_id": [0]}), [review])
    assert not result.timing_consistent.iloc[0]
    assert result.t0_status.iloc[0] == "post"
