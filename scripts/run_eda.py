"""Reproducible exploratory data analysis.

Run from the project root:   python -m scripts.run_eda

Outputs
  data/processed/fact_enrollment.csv   joined, PII-free analytical table
  reports/tables/*.csv                 every table quoted in the paper
  reports/figures/*.png                publication-ready figures
  reports/eda_summary.md               KPIs + key findings (feeds the research paper)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edupro import metrics as M  # noqa: E402
from edupro.config import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, TABLES_DIR, TIER_ORDER  # noqa: E402
from edupro.data import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eda")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
TIER_COLORS = {"Low": "#d1495b", "Mid": "#edae49", "High": "#2e8b57"}


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    tables, report, fact = load_model()
    save_table(report, "data_quality_report")
    fact.to_csv(PROCESSED_DIR / "fact_enrollment.csv", index=False)
    log.info("Joined fact table: %d rows x %d columns", *fact.shape)

    ts, cs = M.teacher_summary(fact), M.course_summary(fact)
    kpi = M.kpis(fact)
    save_table(ts, "instructor_summary")
    save_table(cs, "course_summary")
    save_table(pd.Series(kpi, name="value").rename_axis("kpi").reset_index(), "kpis")

    # ---- 1. Instructor profile -------------------------------------------------
    prof = ts[["TeacherAge", "YearsOfExperience", "TeacherRating"]].describe().T
    save_table(prof.reset_index(), "instructor_profile_describe")
    save_table(tables["teachers"]["Expertise"].value_counts().rename_axis("Expertise")
               .reset_index(name="Instructors"), "expertise_counts")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].hist(ts["TeacherRating"], bins=np.arange(1, 5.25, 0.25), color="#3d5a80", edgecolor="white")
    ax[0].set(title="Instructor rating distribution", xlabel="TeacherRating", ylabel="Instructors")
    ax[1].hist(ts["YearsOfExperience"], bins=12, color="#98c1d9", edgecolor="white")
    ax[1].set(title="Years of experience", xlabel="Years")
    ax[2].hist(ts["TeacherAge"], bins=10, color="#ee6c4d", edgecolor="white")
    ax[2].set(title="Instructor age", xlabel="Age")
    save_fig(fig, "01_instructor_profile")

    # ---- 2. Experience vs performance -----------------------------------------
    corr = M.experience_correlations(ts)
    bins = M.experience_bins(ts)
    bp_t, bp_c = M.experience_breakpoint(ts), M.experience_breakpoint(ts, "AvgCourseRating")
    save_table(corr, "experience_correlations")
    save_table(bins, "experience_bins")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, col, title in [(ax[0], "TeacherRating", "Experience vs TeacherRating"),
                          (ax[1], "AvgCourseRating", "Experience vs CourseRating (enrol.-weighted)")]:
        a.scatter(ts["YearsOfExperience"], ts[col], s=20 + ts["Enrollments"] / 20,
                  c=ts["TeacherRating"], cmap="RdYlGn", edgecolor="k", linewidth=0.3, alpha=0.85)
        fit = M.ols_slope(ts["YearsOfExperience"], ts[col])
        xs = np.linspace(ts["YearsOfExperience"].min(), ts["YearsOfExperience"].max(), 50)
        a.plot(xs, fit["intercept"] + fit["slope"] * xs, "k--", lw=1)
        a.set(title=f"{title}  (slope {fit['slope']:+.3f}/yr)", xlabel="YearsOfExperience", ylabel=col)
    save_fig(fig, "02_experience_vs_rating")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(bins["ExperienceBand"], bins["AvgTeacherRating"], "o-", label="Avg TeacherRating", color="#3d5a80")
    ax.plot(bins["ExperienceBand"], bins["AvgCourseRating"], "s-", label="Avg CourseRating", color="#ee6c4d")
    ax.set(title="Rating by experience band", ylabel="Rating", ylim=(1.5, 4.5))
    ax.legend()
    save_fig(fig, "03_experience_bands")

    # ---- 3. Course quality ----------------------------------------------------
    cat, lvl = M.category_stats(fact), M.level_stats(fact)
    heat, heat_n = M.category_level_matrix(fact)
    gl = M.gender_level_table(fact)
    for df, n in [(cat, "category_stats"), (lvl, "level_stats"), (gl, "gender_level"),
                  (heat.reset_index(), "category_level_matrix")]:
        save_table(df, n)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(heat.values, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns)
    ax.set_yticks(range(heat.shape[0]), heat.index)
    ax.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v, n = heat.iat[i, j], heat_n.iat[i, j]
            ax.text(j, i, "-" if pd.isna(v) else f"{v:.2f}\n(n={int(n)})", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Mean CourseRating")
    ax.set_title("Course quality: category x level")
    save_fig(fig, "04_category_level_heatmap")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    c2 = cat.sort_values("AvgCourseRating")
    ax.barh(c2["CourseCategory"], c2["AvgCourseRating"], xerr=c2["RatingStd"], color="#5c80bc",
            error_kw={"alpha": 0.5})
    ax.axvline(cs["CourseRating"].mean(), color="k", ls="--", lw=1)
    ax.set(title="Mean course rating by category (bars = +/-1 SD across courses)", xlabel="CourseRating")
    save_fig(fig, "05_category_ratings")

    # ---- 4. Instructor impact -------------------------------------------------
    tier = M.tier_comparison(fact)
    save_table(tier, "tier_comparison")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = [TIER_COLORS[t] for t in TIER_ORDER]
    ax[0].bar(tier["RatingTier"], tier["AvgCourseRating"], color=colors)
    ax[0].set(title="Avg course rating by instructor tier", ylim=(0, 5))
    ax[1].bar(tier["RatingTier"], tier["MeanEnrollmentsPerInstructor"], color=colors)
    ax[1].set(title="Mean enrolments per instructor")
    ax[2].bar(tier["RatingTier"], tier["EnrollmentInfluenceRatio"], color=colors)
    ax[2].axhline(1, color="k", ls="--", lw=1)
    ax[2].set(title="Enrollment Influence Ratio (1 = proportional)")
    save_fig(fig, "06_instructor_tiers")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(ts["TeacherRating"], ts["Enrollments"], c=[TIER_COLORS[t] for t in ts["RatingTier"]],
               edgecolor="k", linewidth=0.3)
    ax.set(yscale="log", title="TeacherRating vs enrolments (log scale)", xlabel="TeacherRating",
           ylabel="Enrolments")
    save_fig(fig, "07_rating_vs_enrollment")

    # ---- 5. Expertise ---------------------------------------------------------
    exp = M.expertise_stats(fact)
    dep = M.category_dependence(fact)
    save_table(exp, "expertise_stats")
    save_table(dep, "category_dependence")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    e2 = exp.sort_values("AvgTeacherRating")
    y = np.arange(len(e2))
    ax.barh(y - 0.2, e2["AvgTeacherRating"], 0.4, label="Avg TeacherRating", color="#3d5a80")
    ax.barh(y + 0.2, e2["AvgCourseRating"], 0.4, label="Avg CourseRating", color="#ee6c4d")
    ax.set_yticks(y, [f"{n}{' *' if low else ''}" for n, low in zip(e2["Expertise"], e2["LowSample"])])
    ax.set(title="Expertise area: instructor vs course rating  (* = fewer than 3 instructors)")
    ax.legend()
    save_fig(fig, "08_expertise_performance")

    # ---- Markdown summary -----------------------------------------------------
    rho_t, rho_c = corr.iloc[0], corr.iloc[1]
    pair = dep[dep["CourseCategory"] == "ALL"].iloc[0]
    top = ts.head(5)[["TeacherName", "Expertise", "YearsOfExperience", "TeacherRating", "Enrollments"]]
    low = ts.tail(5)[["TeacherName", "Expertise", "YearsOfExperience", "TeacherRating", "Enrollments"]]
    hi = tier.set_index("RatingTier")
    lines = [
        "# EduPro EDA summary (auto-generated)", "",
        "## Data integrity",
        f"- Joined {len(fact):,} enrolments to {kpi['n_instructors']} instructors, {kpi['n_courses']} courses "
        f"and {fact['UserID'].nunique():,} learners with zero orphan keys and zero row fan-out.",
        "- **Teachers <-> Courses is many-to-many**: the only link is the Transactions table, and each course "
        "is delivered by 7-30 different instructors. `CourseRating` is therefore a property of the course; "
        "instructor-level course quality is the enrolment-weighted mean of the courses an instructor sold.", "",
        "## KPIs",
        f"- Average Teacher Rating: **{kpi['avg_teacher_rating']:.2f}**",
        f"- Average Course Rating: **{kpi['avg_course_rating']:.2f}** (enrolment-weighted {kpi['avg_course_rating_weighted']:.2f})",
        f"- Rating Consistency Index: **{kpi['rating_consistency_index']:.2f}** (1 = every course an instructor sells is rated alike)",
        f"- Experience Impact Score: **{kpi['experience_impact_slope']:+.3f} rating pts per extra year** "
        f"(Spearman rho = {kpi['experience_impact_rho']:.2f}, p = {kpi['experience_impact_p']:.3g})",
        f"- Enrollment Influence Ratio (High tier): **{kpi['enrollment_influence_ratio']:.2f}x** "
        f"({hi.loc['High', 'ShareOfEnrollments']:.0%} of enrolments from {hi.loc['High', 'ShareOfInstructors']:.0%} of instructors)", "",
        "## Findings",
        f"1. **Distribution** - instructor ratings range {ts['TeacherRating'].min():.2f}-{ts['TeacherRating'].max():.2f}, "
        f"mean {ts['TeacherRating'].mean():.2f}, SD {ts['TeacherRating'].std():.2f}; Gini {kpi['teacher_rating_gini']:.2f}. "
        f"Enrolments are far more concentrated (Gini {kpi['enrollment_gini']:.2f}): the top 2 instructors hold "
        f"{ts['Enrollments'].nlargest(2).sum() / ts['Enrollments'].sum():.0%} of all enrolments.",
        f"2. **Experience -> instructor rating**: rho = {rho_t['Spearman rho']:.2f} (p = {rho_t['p-value']:.3g}, n = {int(rho_t['n instructors'])}).",
        f"3. **Experience -> course rating**: rho = {rho_c['Spearman rho']:.2f} (p = {rho_c['p-value']:.2f}) - no detectable link.",
        (f"4. **Threshold**: a single-knot fit places the change point at ~{bp_t['knot_years']} years "
         f"(slope {bp_t['slope_before']:+.2f}/yr before, {bp_t['slope_after']:+.2f}/yr after)." if bp_t else
         "4. Not enough data for a threshold fit."),
        f"5. **TeacherRating vs CourseRating** (instructor-course pairs, n = {int(pair['Pairs'])}): rho = {pair['Spearman rho']:.2f} "
        f"(p = {pair['p-value']:.2f}) - statistically independent in this dataset.",
        f"6. **Tiers**: course ratings are flat across tiers ({hi.loc['Low', 'AvgCourseRating']:.2f} / "
        f"{hi.loc['Mid', 'AvgCourseRating']:.2f} / {hi.loc['High', 'AvgCourseRating']:.2f} for Low/Mid/High) while "
        f"High-tier instructors average {hi.loc['High', 'MeanEnrollmentsPerInstructor']:.0f} enrolments vs "
        f"{hi.loc['Mid', 'MeanEnrollmentsPerInstructor']:.0f} for Mid - driven by a handful of very large instructors "
        f"(medians: {hi.loc['High', 'MedianEnrollmentsPerInstructor']:.0f} vs {hi.loc['Mid', 'MedianEnrollmentsPerInstructor']:.0f}).",
        f"7. **Categories**: best {cat.iloc[0]['CourseCategory']} ({cat.iloc[0]['AvgCourseRating']:.2f}), weakest "
        f"{cat.iloc[-1]['CourseCategory']} ({cat.iloc[-1]['AvgCourseRating']:.2f}); most consistent "
        f"{cat.sort_values('ConsistencyIndex').iloc[-1]['CourseCategory']}, least consistent "
        f"{cat.sort_values('ConsistencyIndex').iloc[0]['CourseCategory']}. Levels: "
        + ", ".join(f"{r.CourseLevel} {r.AvgCourseRating:.2f}" for r in lvl.itertuples()) + ".",
        "", "## Top 5 instructors", top.to_markdown(index=False) if _has_tabulate() else top.to_string(index=False),
        "", "## Bottom 5 instructors", low.to_markdown(index=False) if _has_tabulate() else low.to_string(index=False),
        "", "## Caveats",
        "- n = 60 instructors and 60 courses: treat all tests as exploratory.",
        "- Category x level cells hold 1-4 courses; read the heatmap as descriptive, not inferential.",
        "- Expertise areas with fewer than 3 instructors are flagged low-sample.",
    ]
    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("EDA complete -> %s", REPORTS_DIR)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
