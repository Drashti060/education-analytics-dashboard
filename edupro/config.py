"""Central configuration: paths, schemas and analytical constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Raw files are matched by glob so "EduPro Online Platform.xlsx - Teachers.csv",
# "EduPro_Online_Platform_xlsx_-_Teachers.csv" and a plain "Teachers.csv" all work.
SCHEMAS: dict[str, dict] = {
    "teachers": {
        "pattern": "*Teachers*.csv",
        "key": "TeacherID",
        "required": ["TeacherID", "TeacherName", "Age", "Gender", "Expertise",
                     "YearsOfExperience", "TeacherRating"],
    },
    "courses": {
        "pattern": "*Courses*.csv",
        "key": "CourseID",
        "required": ["CourseID", "CourseName", "CourseCategory", "CourseType",
                     "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"],
    },
    "transactions": {
        "pattern": "*Transactions*.csv",
        "key": "TransactionID",
        "required": ["TransactionID", "UserID", "CourseID", "TransactionDate",
                     "Amount", "PaymentMethod", "TeacherID"],
    },
    "users": {
        "pattern": "*Users*.csv",
        "key": "UserID",
        "required": ["UserID", "Age", "Gender"],  # UserName / Email are optional (PII)
    },
}

# PII is dropped at ingest and never reaches the analytics layer, the UI or exports.
PII_COLUMNS = ["UserName", "Email"]

DATE_FORMAT = "%d/%m/%Y"            # source dates look like 25/10/2025 and 3/1/2025
RATING_MIN, RATING_MAX = 0.0, 5.0   # valid range for every rating field
MAX_RATING_STD = 2.0                # Popoviciu bound: max std of a variable on [1, 5]

# Instructor rating tiers (adjustable in the UI)
TIER_LOW_CUTOFF = 2.5               # rating <  2.5  -> Low
TIER_HIGH_CUTOFF = 3.75             # rating >= 3.75 -> High, otherwise Mid
TIER_ORDER = ["Low", "Mid", "High"]

EXPERIENCE_BINS = [0, 2, 5, 8, 12, float("inf")]
EXPERIENCE_LABELS = ["1-2 yrs", "3-5 yrs", "6-8 yrs", "9-12 yrs", "13+ yrs"]

LEARNER_AGE_BINS = [0, 17, 24, 34, 200]
LEARNER_AGE_LABELS = ["Under 18", "18-24", "25-34", "35+"]

MIN_INSTRUCTORS_FOR_CONFIDENCE = 3  # expertise areas below this are flagged "low sample"
