import pandas as pd
import pytest

from edupro.data import DataValidationError, assert_valid, assign_tier, build_fact, validate


def test_join_preserves_grain_and_adds_dimensions(tables, fact):
    assert len(fact) == len(tables["transactions"])
    for col in ["CourseRating", "TeacherRating", "LearnerAge", "ExpertiseMatch", "RatingTier"]:
        assert col in fact.columns
    assert fact["TransactionDate"].dt.month.tolist() == [1, 2, 3, 4, 5, 6]  # dd/mm/YYYY parsed


def test_duplicate_dimension_key_is_rejected(tables):
    tables["courses"] = pd.concat([tables["courses"], tables["courses"].iloc[[0]]])
    with pytest.raises(Exception):
        build_fact(tables)


def test_validation_flags_orphans(tables):
    tables["transactions"].loc[0, "TeacherID"] = "T999"
    rep = validate(tables)
    assert (rep["Status"] == "FAIL").any()
    with pytest.raises(DataValidationError):
        assert_valid(rep)


def test_validation_passes_on_clean_data(tables):
    assert not (validate(tables)["Status"] == "FAIL").any()


def test_assign_tier_boundaries():
    s = pd.Series([2.49, 2.5, 3.74, 3.75])
    assert assign_tier(s, 2.5, 3.75).astype(str).tolist() == ["Low", "Mid", "Mid", "High"]
    with pytest.raises(ValueError):
        assign_tier(s, 4, 3)
