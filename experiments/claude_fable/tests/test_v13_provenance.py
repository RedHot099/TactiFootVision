import pytest

from xgpoc.v13.provenance import snapshot, verify_sources


def test_mid_run_edit_cannot_be_signed_as_executed_code(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text("old code")
    recorded = snapshot([source])
    verify_sources(recorded)
    source.write_text("new code")
    with pytest.raises(ValueError, match="Source changed"):
        verify_sources(recorded)
