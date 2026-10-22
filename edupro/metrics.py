"""Analytical layer: KPIs and aggregations. Pure functions, no I/O, no Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    EXPERIENCE_BINS, EXPERIENCE_LABELS, MAX_RATING_STD, MIN_INSTRUCTORS_FOR_CONFIDENCE,
    TIER_ORDER,
)

TEACHER_ATTRS = ["TeacherName", "TeacherAge", "TeacherGender", "Expertise",
                 "YearsOfExperience", "TeacherRating"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gini(values) -> float:
    """Gini coefficient (0 = perfectly even, 1 = all mass on one unit)."""
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def spearman(x, y) -> dict:
    """Spearman rho with p-value and n; NaN-safe."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return {"rho": float("nan"), "p": float("nan"), "n": len(d)}
    rho, p = stats.spearmanr(d["x"], d["y"])
    return {"rho": float(rho), "p": float(p), "n": len(d)}


def ols_slope(x, y) -> dict:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3 or d["x"].nunique() < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    r = stats.linregress(d["x"], d["y"])
    return {"slope": float(r.slope), "intercept": float(r.intercept), "r2": float(r.rvalue ** 2)}


# --------------------------------------------------------------------------- #
# Instructor-level table
# --------------------------------------------------------------------------- #
def teacher_summary(fact: pd.DataFrame) -> pd.DataFrame:
    """One row per instructor that appears in `fact`.

    AvgCourseRating is the enrolment-weighted mean rating of the courses the
    instructor was enrolled through. RatingConsistencyIndex = 1 - std/2, where
    std is the spread of those course ratings (1 = every course rated the same).
    """
    g = fact.groupby("TeacherID").agg(
        Enrollments=("TransactionID", "count"),
        Courses=("CourseID", "nunique"),
        Revenue=("Amount", "sum"),
        AvgCourseRating=("CourseRating", "mean"),
        CourseRatingStd=("CourseRating", "std"),
        ExpertiseMatchShare=("ExpertiseMatch", "mean"),
    )
    attrs = fact.groupby("TeacherID")[TEACHER_ATTRS].first()
    ts = attrs.join(g).reset_index()
    ts["RatingConsistencyIndex"] = (1 - ts["CourseRatingStd"] / MAX_RATING_STD).clip(0, 1)
    ts["RatingTier"] = fact.groupby("TeacherID")["RatingTier"].first().reindex(ts["TeacherID"]).values
    ts["Rank"] = ts["TeacherRating"].rank(ascending=False, method="min").astype(int)
    return ts.sort_values(["TeacherRating", "Enrollments"], ascending=False).reset_index(drop=True)


def course_summary(fact: pd.DataFrame) -> pd.DataFrame:
    """One row per course (the correct grain for course-quality statistics)."""
    g = fact.groupby("CourseID").agg(
        Enrollments=("TransactionID", "count"),
        Instructors=("TeacherID", "nunique"),
    )
    attrs = fact.groupby("CourseID")[["CourseName", "CourseCategory", "CourseLevel", "CourseType",
                                      "CoursePrice", "CourseDuration", "CourseRating"]].first()
    return attrs.join(g).reset_index()


def teacher_course_pairs(fact: pd.DataFrame) -> pd.DataFrame:
    """Unique (instructor, course) pairs - least pseudo-replicated grain for
    teacher-rating vs course-rating analysis."""
    cols = ["TeacherID", "CourseID", "TeacherRating", "CourseRating", "Expertise",
            "CourseCategory", "CourseLevel", "TeacherGender"]
    pairs = fact.drop_duplicates(["TeacherID", "CourseID"])[cols].copy()
    pairs["Enrollments"] = fact.groupby(["TeacherID", "CourseID"]).size() \
        .reindex(pd.MultiIndex.from_frame(pairs[["TeacherID", "CourseID"]])).values
    return pairs


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def kpis(fact: pd.DataFrame) -> dict:
    ts = teacher_summary(fact)
    cs = course_summary(fact)
    exp = spearman(ts["YearsOfExperience"], ts["TeacherRating"])
    fit = ols_slope(ts["YearsOfExperience"], ts["TeacherRating"])

    total_enr = ts["Enrollments"].sum()
    high = ts[ts["RatingTier"] == "High"]
    if len(high) and total_enr:
        eir = (high["Enrollments"].sum() / total_enr) / (len(high) / len(ts))
    else:
        eir = float("nan")

    return {
        "avg_teacher_rating": float(ts["TeacherRating"].mean()),
        "avg_course_rating": float(cs["CourseRating"].mean()),
        "avg_course_rating_weighted": float(fact["CourseRating"].mean()),
        "rating_consistency_index": float(ts["RatingConsistencyIndex"].mean()),
        "experience_impact_slope": fit["slope"],           # rating points per extra year
        "experience_impact_rho": exp["rho"],
        "experience_impact_p": exp["p"],
        "enrollment_influence_ratio": float(eir),
        "n_instructors": int(len(ts)),
        "n_courses": int(len(cs)),
        "n_enrollments": int(len(fact)),
        "enrollment_gini": gini(ts["Enrollments"]),
        "teacher_rating_gini": gini(ts["TeacherRating"]),
    }


# --------------------------------------------------------------------------- #
# Experience vs performance
# --------------------------------------------------------------------------- #
def experience_correlations(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, label in [("TeacherRating", "YearsOfExperience vs TeacherRating"),
                          ("AvgCourseRating", "YearsOfExperience vs CourseRating (enrol.-weighted)")]:
        sp = spearman(ts["YearsOfExperience"], ts[target])
        pe = ts[["YearsOfExperience", target]].dropna().corr().iloc[0, 1] if len(ts) > 2 else np.nan
        rows.append({"Relationship": label, "Spearman rho": sp["rho"], "p-value": sp["p"],
                     "Pearson r": pe, "n instructors": sp["n"]})
    return pd.DataFrame(rows)


def experience_bins(ts: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(ts["YearsOfExperience"], bins=EXPERIENCE_BINS, labels=EXPERIENCE_LABELS)
    out = ts.groupby(b, observed=True).agg(
        Instructors=("TeacherID", "count"),
        AvgTeacherRating=("TeacherRating", "mean"),
        AvgCourseRating=("AvgCourseRating", "mean"),
        AvgEnrollments=("Enrollments", "mean"),
    ).reset_index().rename(columns={"YearsOfExperience": "ExperienceBand"})
    out["TeacherRatingChange"] = out["AvgTeacherRating"].diff()
    return out


def experience_breakpoint(ts: pd.DataFrame, y: str = "TeacherRating") -> dict:
    """Single-knot piecewise-linear fit; the knot minimising SSE is the experience
    level beyond which the marginal return of an extra year changes (diminishing
    returns if slope_after < slope_before). Indicative only: small n."""
    d = ts[["YearsOfExperience", y]].dropna()
    if len(d) < 12:
        return {}
    x, v = d["YearsOfExperience"].to_numpy(float), d[y].to_numpy(float)
    best = None
    for k in range(int(np.percentile(x, 15)) + 1, int(np.percentile(x, 85)) + 1):
        X = np.column_stack([np.ones_like(x), x, np.maximum(x - k, 0)])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        sse = float(((v - X @ beta) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"knot_years": k, "slope_before": float(beta[1]),
                    "slope_after": float(beta[1] + beta[2]), "sse": sse}
    if best:
        best["diminishing_returns"] = bool(
            best["slope_before"] > 0.02 and best["slope_after"] < best["slope_before"] * 0.5
        )
    return best or {}


# --------------------------------------------------------------------------- #
# Course quality
# --------------------------------------------------------------------------- #
def category_stats(fact: pd.DataFrame) -> pd.DataFrame:
    cs = course_summary(fact)
    out = cs.groupby("CourseCategory").agg(
        Courses=("CourseID", "count"),
        AvgCourseRating=("CourseRating", "mean"),
        RatingStd=("CourseRating", "std"),
        MinRating=("CourseRating", "min"),
        MaxRating=("CourseRating", "max"),
        Enrollments=("Enrollments", "sum"),
    ).reset_index()
    out["ConsistencyIndex"] = (1 - out["RatingStd"].fillna(0) / MAX_RATING_STD).clip(0, 1)
    return out.sort_values("AvgCourseRating", ascending=False).reset_index(drop=True)


def level_stats(fact: pd.DataFrame) -> pd.DataFrame:
    cs = course_summary(fact)
    return (cs.groupby("CourseLevel")
              .agg(Courses=("CourseID", "count"), AvgCourseRating=("CourseRating", "mean"),
                   RatingStd=("CourseRating", "std"), Enrollments=("Enrollments", "sum"))
              .reset_index())


def category_level_matrix(fact: pd.DataFrame, weighted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean CourseRating (rows: category, cols: level) plus the number of courses per cell."""
    cs = course_summary(fact)
    n = cs.pivot_table(index="CourseCategory", columns="CourseLevel", values="CourseID",
                       aggfunc="count")
    if weighted:
        mean = fact.pivot_table(index="CourseCategory", columns="CourseLevel",
                                values="CourseRating", aggfunc="mean")
    else:
        mean = cs.pivot_table(index="CourseCategory", columns="CourseLevel",
                              values="CourseRating", aggfunc="mean")
    order = [c for c in ["Beginner", "Intermediate", "Advanced"] if c in mean.columns]
    return mean[order], n.reindex(index=mean.index, columns=order)


def gender_level_table(fact: pd.DataFrame) -> pd.DataFrame:
    """Instructor gender x course level (enrolment-weighted course rating)."""
    return (fact.groupby(["TeacherGender", "CourseLevel"])
                .agg(AvgCourseRating=("CourseRating", "mean"), Enrollments=("TransactionID", "count"),
                     Instructors=("TeacherID", "nunique"))
                .reset_index())


# --------------------------------------------------------------------------- #
# Instructor impact
# --------------------------------------------------------------------------- #
def tier_comparison(fact: pd.DataFrame) -> pd.DataFrame:
    ts = teacher_summary(fact)
    out = ts.groupby("RatingTier", observed=False).agg(
        Instructors=("TeacherID", "count"),
        AvgTeacherRating=("TeacherRating", "mean"),
        AvgCourseRating=("AvgCourseRating", "mean"),
        TotalEnrollments=("Enrollments", "sum"),
        MeanEnrollmentsPerInstructor=("Enrollments", "mean"),
        MedianEnrollmentsPerInstructor=("Enrollments", "median"),
        Revenue=("Revenue", "sum"),
    ).reindex(TIER_ORDER).reset_index()
    tot_i, tot_e = out["Instructors"].sum(), out["TotalEnrollments"].sum()
    out["ShareOfInstructors"] = out["Instructors"] / tot_i if tot_i else np.nan
    out["ShareOfEnrollments"] = out["TotalEnrollments"] / tot_e if tot_e else np.nan
    out["EnrollmentInfluenceRatio"] = out["ShareOfEnrollments"] / out["ShareOfInstructors"]
    return out


# --------------------------------------------------------------------------- #
# Expertise
# --------------------------------------------------------------------------- #
def expertise_stats(fact: pd.DataFrame) -> pd.DataFrame:
    ts = teacher_summary(fact)
    out = ts.groupby("Expertise").agg(
        Instructors=("TeacherID", "count"),
        AvgTeacherRating=("TeacherRating", "mean"),
        AvgCourseRating=("AvgCourseRating", "mean"),
        AvgExperience=("YearsOfExperience", "mean"),
        Enrollments=("Enrollments", "sum"),
        ExpertiseMatchShare=("ExpertiseMatchShare", "mean"),
    ).reset_index()
    platform = ts["TeacherRating"].mean()
    share = out["Enrollments"] / out["Enrollments"].sum()
    out["LowSample"] = out["Instructors"] < MIN_INSTRUCTORS_FOR_CONFIDENCE
    # Training-need score: how far below the platform benchmark, weighted by demand.
    out["GapScore"] = (platform - out["AvgTeacherRating"]).clip(lower=0) * share * 100
    return out.sort_values("AvgTeacherRating", ascending=False).reset_index(drop=True)


def category_dependence(fact: pd.DataFrame) -> pd.DataFrame:
    """Per course category: how strongly does the instructor's rating track the course
    rating? (Spearman at instructor-course-pair grain.)  Overall row included."""
    pairs = teacher_course_pairs(fact)
    rows = []
    for cat, d in pairs.groupby("CourseCategory"):
        sp = spearman(d["TeacherRating"], d["CourseRating"])
        rows.append({"CourseCategory": cat, "Spearman rho": sp["rho"], "p-value": sp["p"],
                     "Pairs": sp["n"]})
    sp = spearman(pairs["TeacherRating"], pairs["CourseRating"])
    rows.append({"CourseCategory": "ALL", "Spearman rho": sp["rho"], "p-value": sp["p"],
                 "Pairs": sp["n"]})
    return pd.DataFrame(rows).sort_values("Spearman rho", ascending=False).reset_index(drop=True)


def apply_filters(fact: pd.DataFrame, *, expertise=None, categories=None, levels=None,
                  teacher_rating=None, course_rating=None) -> pd.DataFrame:
    """Row-level filter on the joined fact table (each argument optional)."""
    m = pd.Series(True, index=fact.index)
    if expertise:
        m &= fact["Expertise"].isin(expertise)
    if categories:
        m &= fact["CourseCategory"].isin(categories)
    if levels:
        m &= fact["CourseLevel"].isin(levels)
    if teacher_rating:
        m &= fact["TeacherRating"].between(*teacher_rating)
    if course_rating:
        m &= fact["CourseRating"].between(*course_rating)
    return fact[m]
