"""EduPro - Instructor Performance & Course Quality Evaluation (Streamlit dashboard).

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edupro import metrics as M
from edupro.config import (
    TIER_HIGH_CUTOFF, TIER_LOW_CUTOFF, TIER_ORDER,
)
from edupro.data import assign_tier, load_model

st.set_page_config(page_title="EduPro | Instructor Performance", page_icon="🎓", layout="wide")

TIER_COLORS = {"Low": "#d1495b", "Mid": "#edae49", "High": "#2e8b57"}
LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]


# --------------------------------------------------------------------------- #
# Data (cached; PII is dropped inside load_raw)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and joining datasets...")
def get_data():
    return load_model()


def fmt(v: float, nd: int = 2, suffix: str = "") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}{suffix}"


def trend_trace(x: pd.Series, y: pd.Series) -> go.Scatter | None:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3 or d["x"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(d["x"], d["y"], 1)
    xs = np.linspace(d["x"].min(), d["x"].max(), 50)
    return go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", name="OLS trend",
                      line=dict(color="black", dash="dash", width=1.5))


try:
    tables, quality_report, fact_all = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the datasets: {exc}")
    st.info("Place the four CSV files in `data/raw/` (Teachers, Courses, Transactions, Users).")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🎓 EduPro Filters")
sel_expertise = st.sidebar.multiselect("Instructor expertise", sorted(fact_all["Expertise"].unique()))
sel_category = st.sidebar.multiselect("Course category", sorted(fact_all["CourseCategory"].unique()))
sel_level = st.sidebar.multiselect("Course level", LEVEL_ORDER)
t_lo, t_hi = float(np.floor(fact_all["TeacherRating"].min())), float(np.ceil(fact_all["TeacherRating"].max()))
c_lo, c_hi = float(np.floor(fact_all["CourseRating"].min())), float(np.ceil(fact_all["CourseRating"].max()))
sel_trating = st.sidebar.slider("Instructor rating range", t_lo, t_hi, (t_lo, t_hi), 0.05)
sel_crating = st.sidebar.slider("Course rating range", c_lo, c_hi, (c_lo, c_hi), 0.05)
with st.sidebar.expander("Instructor tier definitions"):
    low_cut, high_cut = st.slider("Low < first cut-off <= Mid < second cut-off <= High", 1.0, 5.0,
                                  (TIER_LOW_CUTOFF, TIER_HIGH_CUTOFF), 0.05)
min_enr = st.sidebar.number_input("Min. enrolments per instructor (leaderboard)", 0, 500, 0, 10)
st.sidebar.caption("Empty selections mean 'all'.")

fact = M.apply_filters(fact_all, expertise=sel_expertise, categories=sel_category, levels=sel_level,
                        teacher_rating=sel_trating, course_rating=sel_crating)
if fact.empty:
    st.warning("No enrolments match the current filters. Widen the selection.")
    st.stop()
fact = fact.assign(RatingTier=assign_tier(fact["TeacherRating"], low_cut, high_cut))

ts = M.teacher_summary(fact)
kp = M.kpis(fact)

st.title("Instructor Performance & Course Quality Evaluation")
st.caption(f"{kp['n_enrollments']:,} enrolments · {kp['n_instructors']} instructors · "
           f"{kp['n_courses']} courses (filtered view of {len(fact_all):,} total enrolments)")

tab_ov, tab_lb, tab_exp, tab_cq, tab_imp, tab_ex, tab_dq = st.tabs([
    "Overview", "Leaderboard", "Experience vs Rating", "Course Quality",
    "Instructor Impact", "Expertise", "Data & Joins"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_ov:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Teacher Rating", fmt(kp["avg_teacher_rating"]), help="Mean over instructors")
    c2.metric("Avg Course Rating", fmt(kp["avg_course_rating"]),
              help=f"Mean over courses. Enrolment-weighted: {fmt(kp['avg_course_rating_weighted'])}")
    c3.metric("Rating Consistency Index", fmt(kp["rating_consistency_index"]),
              help="1 - SD(course ratings an instructor sells) / 2, averaged over instructors. 1 = perfectly uniform.")
    c4.metric("Experience Impact Score", fmt(kp["experience_impact_slope"], 3, " /yr"),
              help=f"OLS slope of TeacherRating on YearsOfExperience. Spearman rho = {fmt(kp['experience_impact_rho'])}, "
                   f"p = {fmt(kp['experience_impact_p'], 4)}")
    c5.metric("Enrollment Influence Ratio", fmt(kp["enrollment_influence_ratio"], 2, "x"),
              help="Share of enrolments captured by High-tier instructors / their share of instructors. >1 = outsized demand.")

    left, right = st.columns(2)
    fig = px.histogram(ts, x="TeacherRating", nbins=16, color="RatingTier", color_discrete_map=TIER_COLORS,
                       category_orders={"RatingTier": TIER_ORDER}, title="Distribution of instructor ratings")
    fig.update_layout(bargap=0.05, yaxis_title="Instructors")
    left.plotly_chart(fig)
    fig = px.histogram(ts, x="CourseRatingStd", nbins=15, title="Spread of course ratings within instructors",
                       labels={"CourseRatingStd": "SD of course ratings sold"})
    fig.update_layout(yaxis_title="Instructors")
    right.plotly_chart(fig)

    g1, g2, g3 = st.columns(3)
    g1.metric("Gini - instructor ratings", fmt(kp["teacher_rating_gini"]),
              help="0 = all instructors rated alike; higher = more unequal.")
    g2.metric("Gini - enrolments", fmt(kp["enrollment_gini"]),
              help="How concentrated demand is across instructors (0 even, 1 = one instructor).")
    top2 = ts["Enrollments"].nlargest(2).sum() / ts["Enrollments"].sum()
    g3.metric("Top-2 instructors' enrolment share", f"{top2:.0%}")

# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
with tab_lb:
    lb = ts[ts["Enrollments"] >= min_enr].copy()
    mode = st.radio("Show", ["Top performers", "Lowest performers", "All"], horizontal=True)
    max_rows = max(5, len(lb))
    if max_rows > 5:
        n_show = st.slider("Rows", 5, max_rows, min(15, max_rows), 1)
    else:
        n_show = len(lb)
    if mode == "Top performers":
        lb = lb.nlargest(n_show, ["TeacherRating", "Enrollments"])
    elif mode == "Lowest performers":
        lb = lb.nsmallest(n_show, ["TeacherRating", "Enrollments"])
    else:
        lb = lb.head(n_show)
    cols = ["Rank", "TeacherName", "Expertise", "YearsOfExperience", "TeacherRating", "RatingTier",
            "AvgCourseRating", "RatingConsistencyIndex", "Courses", "Enrollments", "Revenue"]
    st.dataframe(
        lb[cols], hide_index=True,
        column_config={
            "TeacherRating": st.column_config.ProgressColumn("Teacher rating", min_value=0, max_value=5, format="%.2f"),
            "AvgCourseRating": st.column_config.NumberColumn("Avg course rating", format="%.2f"),
            "RatingConsistencyIndex": st.column_config.NumberColumn("Consistency", format="%.2f"),
            "Revenue": st.column_config.NumberColumn(format="%.0f"),
            "YearsOfExperience": st.column_config.NumberColumn("Experience (yrs)"),
        })
    st.download_button("Download instructor table (CSV)", ts.to_csv(index=False).encode(),
                       "instructor_summary.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Experience vs rating
# --------------------------------------------------------------------------- #
with tab_exp:
    a, b = st.columns(2)
    for col, target, title in [(a, "TeacherRating", "Experience vs instructor rating"),
                               (b, "AvgCourseRating", "Experience vs course rating (enrolment-weighted)")]:
        fig = px.scatter(ts, x="YearsOfExperience", y=target, color="Expertise", size="Enrollments",
                         hover_name="TeacherName", title=title, size_max=28)
        tr = trend_trace(ts["YearsOfExperience"], ts[target])
        if tr:
            fig.add_trace(tr)
        col.plotly_chart(fig)
    st.subheader("Correlation evidence")
    st.dataframe(M.experience_correlations(ts), hide_index=True, column_config={
        "Spearman rho": st.column_config.NumberColumn(format="%.3f"),
        "p-value": st.column_config.NumberColumn(format="%.4f"),
        "Pearson r": st.column_config.NumberColumn(format="%.3f")})

    bins = M.experience_bins(ts)
    left, right = st.columns([3, 2])
    if not bins.empty:
        fig = go.Figure()
        fig.add_scatter(x=bins["ExperienceBand"], y=bins["AvgTeacherRating"], mode="lines+markers", name="Avg TeacherRating")
        fig.add_scatter(x=bins["ExperienceBand"], y=bins["AvgCourseRating"], mode="lines+markers", name="Avg CourseRating")
        fig.update_layout(title="Rating by experience band (diminishing-returns check)", yaxis_title="Rating")
        left.plotly_chart(fig)
    bp = M.experience_breakpoint(ts)
    if bp:
        verdict = "diminishing returns" if bp["diminishing_returns"] else "no clear plateau"
        right.info(f"**Change point ≈ {bp['knot_years']} years** ({verdict}). Each extra year is associated with "
                   f"{bp['slope_before']:+.2f} rating pts before and {bp['slope_after']:+.2f} after. "
                   "Indicative only: small sample.")
    else:
        right.info("Not enough instructors in the current filter to estimate a threshold.")

# --------------------------------------------------------------------------- #
# Course quality
# --------------------------------------------------------------------------- #
with tab_cq:
    weighted = st.toggle("Weight heatmap by enrolments", value=False,
                         help="Off: each course counts once. On: each enrolment counts once.")
    mean_m, n_m = M.category_level_matrix(fact, weighted)
    fig = px.imshow(mean_m, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=1, zmax=5, aspect="auto",
                    title="Mean CourseRating: category x level", labels=dict(color="Rating"))
    st.plotly_chart(fig)
    st.caption("Cells contain only 1-4 courses each; read as descriptive. Blank = no course.")
    with st.expander("Courses per cell"):
        st.dataframe(n_m.fillna(0).astype(int))

    left, right = st.columns(2)
    cat = M.category_stats(fact)
    fig = px.bar(cat.sort_values("AvgCourseRating"), x="AvgCourseRating", y="CourseCategory", orientation="h",
                 error_x="RatingStd", color="ConsistencyIndex", color_continuous_scale="Blues",
                 title="Category quality (error bar = SD across courses)")
    left.plotly_chart(fig)
    lvl = M.level_stats(fact)
    fig = px.bar(lvl, x="CourseLevel", y="AvgCourseRating", error_y="RatingStd", text_auto=".2f",
                 category_orders={"CourseLevel": LEVEL_ORDER}, title="Course rating by level")
    right.plotly_chart(fig)

    gl = M.gender_level_table(fact)
    fig = px.bar(gl, x="CourseLevel", y="AvgCourseRating", color="TeacherGender", barmode="group", text_auto=".2f",
                 category_orders={"CourseLevel": LEVEL_ORDER}, title="Instructor gender x course level")
    st.plotly_chart(fig)
    st.dataframe(cat, hide_index=True)

# --------------------------------------------------------------------------- #
# Instructor impact
# --------------------------------------------------------------------------- #
with tab_imp:
    tc = M.tier_comparison(fact)
    st.dataframe(tc, hide_index=True, column_config={
        "ShareOfInstructors": st.column_config.NumberColumn(format="percent"),
        "ShareOfEnrollments": st.column_config.NumberColumn(format="percent"),
        "AvgTeacherRating": st.column_config.NumberColumn(format="%.2f"),
        "AvgCourseRating": st.column_config.NumberColumn(format="%.2f"),
        "EnrollmentInfluenceRatio": st.column_config.NumberColumn(format="%.2f"),
        "MeanEnrollmentsPerInstructor": st.column_config.NumberColumn(format="%.0f"),
        "MedianEnrollmentsPerInstructor": st.column_config.NumberColumn(format="%.0f"),
        "Revenue": st.column_config.NumberColumn(format="%.0f")})
    a, b, c = st.columns(3)
    a.plotly_chart(px.box(ts, x="RatingTier", y="AvgCourseRating", color="RatingTier", points="all",
                          color_discrete_map=TIER_COLORS, category_orders={"RatingTier": TIER_ORDER},
                          title="Course rating by instructor tier").update_layout(showlegend=False))
    b.plotly_chart(px.bar(tc, x="RatingTier", y="MeanEnrollmentsPerInstructor", color="RatingTier",
                          color_discrete_map=TIER_COLORS, title="Mean enrolments per instructor",
                          text_auto=".0f").update_layout(showlegend=False))
    c.plotly_chart(px.bar(tc, x="RatingTier", y="MedianEnrollmentsPerInstructor", color="RatingTier",
                          color_discrete_map=TIER_COLORS, title="Median enrolments per instructor",
                          text_auto=".0f").update_layout(showlegend=False))
    st.caption("Mean vs median: when they diverge, a few very large instructors drive the tier average.")
    fig = px.scatter(ts, x="TeacherRating", y="Enrollments", color="RatingTier", log_y=True,
                     color_discrete_map=TIER_COLORS, hover_name="TeacherName", size="Courses",
                     title="Instructor rating vs enrolments (log scale)")
    st.plotly_chart(fig)
    sp = M.spearman(ts["TeacherRating"], ts["Enrollments"])
    st.write(f"Spearman ρ (rating vs enrolments) = **{fmt(sp['rho'])}**, p = {fmt(sp['p'], 3)}, n = {sp['n']}.")

# --------------------------------------------------------------------------- #
# Expertise
# --------------------------------------------------------------------------- #
with tab_ex:
    ex = M.expertise_stats(fact)
    long = ex.melt(id_vars=["Expertise", "LowSample"], value_vars=["AvgTeacherRating", "AvgCourseRating"],
                   var_name="Metric", value_name="Rating")
    fig = px.bar(long, x="Rating", y="Expertise", color="Metric", barmode="group", orientation="h",
                 title="Expertise area: instructor vs course rating")
    fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig)

    st.subheader("Training-need / expertise-gap view")
    st.caption("GapScore = shortfall of the area's average instructor rating vs the platform average x the area's "
               "share of enrolments x 100. 'Low sample' = fewer than 3 instructors: do not over-interpret.")
    st.dataframe(ex.sort_values("GapScore", ascending=False), hide_index=True, column_config={
        "AvgTeacherRating": st.column_config.NumberColumn(format="%.2f"),
        "AvgCourseRating": st.column_config.NumberColumn(format="%.2f"),
        "AvgExperience": st.column_config.NumberColumn(format="%.1f"),
        "ExpertiseMatchShare": st.column_config.NumberColumn("Teaching in own field", format="percent"),
        "GapScore": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("How much does instructor quality track course quality, by category?")
    dep = M.category_dependence(fact)
    st.dataframe(dep, hide_index=True, column_config={
        "Spearman rho": st.column_config.NumberColumn(format="%.3f"),
        "p-value": st.column_config.NumberColumn(format="%.4f")})
    st.caption("Spearman correlation across instructor-course pairs. Because most courses in a category are sold by "
               "the same instructor pool, within-category correlations can be mechanically near zero.")

# --------------------------------------------------------------------------- #
# Data & joins
# --------------------------------------------------------------------------- #
with tab_dq:
    st.markdown("""
**Join model** - `Transactions` (fact) → `Courses` (m:1 on CourseID) → `Teachers` (m:1 on TeacherID) → `Users` (m:1 on UserID).
Every join is validated many-to-one and the row count is asserted unchanged. `UserName` and `Email` are dropped at ingest.

**Important:** instructors and courses are linked *only* through transactions, and each course is sold under many
different instructors. `CourseRating` is therefore a course attribute; instructor-level course quality is the
enrolment-weighted mean of the courses an instructor sold.
""")
    st.dataframe(quality_report, hide_index=True)
    sizes = pd.DataFrame({"Table": list(tables), "Rows": [len(v) for v in tables.values()],
                          "Columns": [v.shape[1] for v in tables.values()]})
    st.dataframe(sizes, hide_index=True)
    with st.expander("Joined fact table (filtered, first 500 rows)"):
        st.dataframe(fact.head(500), hide_index=True)
    st.download_button("Download joined table (CSV, PII-free)", fact.to_csv(index=False).encode(),
                       "fact_enrollment.csv", "text/csv")