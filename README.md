# EduPro - Instructor Performance & Course Quality Evaluation

Streamlit analytics product built on four CSVs (Teachers, Courses, Transactions, Users).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m scripts.run_eda      # tables, figures, reports/eda_summary.md
pytest -q                      # unit tests + app smoke test
streamlit run app.py           # http://localhost:8501
```

## Architecture
- `edupro/data.py` - ingest (glob-matched filenames, PII dropped), validation report, m:1-checked joins
- `edupro/metrics.py` - KPIs and aggregations (pure functions, unit-tested)
- `app.py` - dashboard;
- `scripts/run_eda.py` - reproducible EDA for the research paper

## Data model caveat
Instructors and courses are linked only through `Transactions`, and every course is sold under 7-30 different
instructors. `CourseRating` is a course attribute; instructor-level course quality = enrolment-weighted mean
rating of the courses they sold.

## KPI definitions
| KPI | Definition |
|---|---|
| Average Teacher Rating | mean `TeacherRating` over instructors |
| Average Course Rating | mean `CourseRating` over courses (enrolment-weighted also shown) |
| Rating Consistency Index | `1 - SD(course ratings sold) / 2`, averaged over instructors (1 = uniform) |
| Experience Impact Score | OLS slope of `TeacherRating` on `YearsOfExperience` (points per year), with Spearman rho |
| Enrollment Influence Ratio | share of enrolments from High-tier instructors / their share of instructors |

## Deploy
**Streamlit Community Cloud**: https://education-analytics-dashboard-maqvty8smu8ncxozmshhus.streamlit.app/
