import math

from edupro import metrics as M


def test_gini_extremes():
    assert M.gini([5, 5, 5, 5]) == 0
    assert M.gini([0, 0, 0, 10]) > 0.7


def test_teacher_summary_counts(fact):
    ts = M.teacher_summary(fact).set_index("TeacherID")
    assert ts["Enrollments"].sum() == len(fact)
    assert ts.loc["T3", "Enrollments"] == 3
    assert 0 <= ts["RatingConsistencyIndex"].min() <= ts["RatingConsistencyIndex"].max() <= 1


def test_kpis_shape(fact):
    k = M.kpis(fact)
    assert k["n_enrollments"] == 6 and k["n_instructors"] == 3 and k["n_courses"] == 2
    assert math.isclose(k["avg_teacher_rating"], (2.0 + 3.0 + 4.5) / 3)


def test_enrollment_influence_ratio(fact):
    tc = M.tier_comparison(fact).set_index("RatingTier")
    assert math.isclose(tc["ShareOfEnrollments"].sum(), 1.0)
    assert tc.loc["High", "Instructors"] == 1  # only T3 (4.5) is High
    assert math.isclose(M.kpis(fact)["enrollment_influence_ratio"], (3 / 6) / (1 / 3))


def test_filters_and_empty_safe(fact):
    sub = M.apply_filters(fact, expertise=["Finance"], levels=["Beginner"])
    assert set(sub["Expertise"]) == {"Finance"} and set(sub["CourseLevel"]) == {"Beginner"}
    assert M.apply_filters(fact, expertise=["Nope"]).empty


def test_spearman_handles_degenerate_input():
    assert math.isnan(M.spearman([1, 1, 1], [1, 2, 3])["rho"])


def test_heatmap_matrix(fact):
    mean, n = M.category_level_matrix(fact)
    assert mean.shape == n.shape
