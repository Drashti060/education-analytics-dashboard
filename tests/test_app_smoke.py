"""End-to-end smoke test of the Streamlit app (skipped if data or streamlit is missing)."""
import pytest

from edupro.config import RAW_DIR

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.mark.skipif(not list(RAW_DIR.glob("*Transactions*.csv")), reason="raw data not present")
def test_app_runs_without_exception():
    at = st_testing.AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 7
