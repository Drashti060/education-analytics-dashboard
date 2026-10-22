"""Ingestion, validation and relational integration of the four EduPro tables.

Entity model (all keys validated):

    Users 1 ──< Transactions >── 1 Courses
                     │
                     └── m:1 ── Teachers

Transactions is the fact table. It is the ONLY place where an instructor is
linked to a course, so the Teachers <-> Courses relationship is many-to-many
and must be read through Transactions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATE_FORMAT, LEARNER_AGE_BINS, LEARNER_AGE_LABELS, PII_COLUMNS, RATING_MAX,
    RATING_MIN, RAW_DIR, SCHEMAS, TIER_HIGH_CUTOFF, TIER_LOW_CUTOFF, TIER_ORDER,
)

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a hard data-integrity rule is violated."""


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _find_csv(raw_dir: Path, pattern: str) -> Path:
    matches = sorted(Path(raw_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {raw_dir}")
    if len(matches) > 1:
        log.warning("Several files match %s; using %s", pattern, matches[0].name)
    return matches[0]


def load_raw(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the four CSVs, normalise text columns and drop PII immediately."""
    tables: dict[str, pd.DataFrame] = {}
    for name, spec in SCHEMAS.items():
        path = _find_csv(Path(raw_dir), spec["pattern"])
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        missing = set(spec["required"]) - set(df.columns)
        if missing:
            raise DataValidationError(f"{path.name}: missing columns {sorted(missing)}")
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].str.strip()
        tables[name] = df
        log.info("Loaded %-12s %6d rows  (%s)", name, len(df), path.name)

    tables["users"] = tables["users"].drop(
        columns=[c for c in PII_COLUMNS if c in tables["users"].columns]
    )
    return tables


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a data-quality report: one row per check with PASS / WARN / FAIL."""
    t, c, x, u = (tables[k] for k in ("teachers", "courses", "transactions", "users"))
    rows: list[tuple[str, str, str]] = []

    def add(check: str, ok: bool, detail: str, severity: str = "FAIL") -> None:
        rows.append((check, "PASS" if ok else severity, detail))

    for name, df in tables.items():
        key = SCHEMAS[name]["key"]
        add(f"{name}: {key} unique", df[key].is_unique,
            f"{df[key].duplicated().sum()} duplicate keys")
        required = SCHEMAS[name]["required"]
        nulls = int(df[required].isna().sum().sum())
        add(f"{name}: no nulls in required columns", nulls == 0, f"{nulls} null cells")

    for col, ref, label in [("CourseID", c, "courses"), ("TeacherID", t, "teachers"),
                            ("UserID", u, "users")]:
        orphans = int((~x[col].isin(ref[col])).sum())
        add(f"transactions.{col} -> {label}", orphans == 0, f"{orphans} orphan rows")

    for df, col in [(t, "TeacherRating"), (c, "CourseRating")]:
        bad = int((~df[col].between(RATING_MIN, RATING_MAX)).sum())
        add(f"{col} within [{RATING_MIN:g}, {RATING_MAX:g}]", bad == 0, f"{bad} out-of-range")

    bad_dates = int(pd.to_datetime(x["TransactionDate"], format=DATE_FORMAT,
                                   errors="coerce").isna().sum())
    add("transactions: dates parse (dd/mm/YYYY)", bad_dates == 0, f"{bad_dates} unparseable")

    add("teachers: experience <= age - 16", bool((t["YearsOfExperience"] <= t["Age"] - 16).all()),
        "career start before age 16" , "WARN")

    free = set(c.loc[c["CourseType"].str.lower() == "free", "CourseID"])
    paid_free = int(((x["CourseID"].isin(free)) & (x["Amount"] > 0)).sum())
    add("free courses carry zero amount", paid_free == 0, f"{paid_free} paid free-course rows", "WARN")

    per_course = x.groupby("CourseID")["TeacherID"].nunique()
    add("course -> instructor mapping is 1:1", bool((per_course == 1).all()),
        f"every course is delivered by {per_course.min()}-{per_course.max()} different "
        f"instructors (median {per_course.median():.0f}); course rating is a COURSE attribute, "
        "not an instructor-specific one", "WARN")

    return pd.DataFrame(rows, columns=["Check", "Status", "Detail"])


def assert_valid(report: pd.DataFrame) -> None:
    failed = report[report["Status"] == "FAIL"]
    if not failed.empty:
        raise DataValidationError("Data validation failed:\n" + failed.to_string(index=False))


# --------------------------------------------------------------------------- #
# Relational integration
# --------------------------------------------------------------------------- #
def assign_tier(ratings: pd.Series, low: float = TIER_LOW_CUTOFF,
                high: float = TIER_HIGH_CUTOFF) -> pd.Series:
    """Low (< low) / Mid (low..high) / High (>= high) as an ordered categorical."""
    if low >= high:
        raise ValueError("low cut-off must be smaller than high cut-off")
    tier = np.select([ratings < low, ratings < high], ["Low", "Mid"], default="High")
    return pd.Series(pd.Categorical(tier, categories=TIER_ORDER, ordered=True),
                     index=ratings.index, name="RatingTier")


def build_fact(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join Transactions with Courses, Teachers and Users (grain: one enrolment).

    Every join is declared many-to-one, so pandas raises if a dimension key is
    duplicated, and the row count is asserted to be unchanged (no fan-out).
    """
    x = tables["transactions"].copy()
    x["TransactionDate"] = pd.to_datetime(x["TransactionDate"], format=DATE_FORMAT, errors="coerce")

    teachers = tables["teachers"].rename(columns={"Age": "TeacherAge", "Gender": "TeacherGender"})
    users = tables["users"].rename(columns={"Age": "LearnerAge", "Gender": "LearnerGender"})

    fact = (
        x.merge(tables["courses"], on="CourseID", how="left", validate="m:1")
         .merge(teachers, on="TeacherID", how="left", validate="m:1")
         .merge(users, on="UserID", how="left", validate="m:1")
    )
    if len(fact) != len(x):
        raise DataValidationError("Join changed the row count (fan-out detected)")

    fact["ExpertiseMatch"] = fact["Expertise"] == fact["CourseCategory"]
    fact["LearnerAgeGroup"] = pd.cut(fact["LearnerAge"], bins=LEARNER_AGE_BINS,
                                     labels=LEARNER_AGE_LABELS).astype("object")
    fact["TransactionMonth"] = fact["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    fact["RatingTier"] = assign_tier(fact["TeacherRating"])
    return fact


def load_model(raw_dir: Path | str = RAW_DIR) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Convenience entry point: tables, validation report, joined fact table."""
    tables = load_raw(raw_dir)
    report = validate(tables)
    assert_valid(report)
    return tables, report, build_fact(tables)


def sanitize_users(raw_dir: Path | str = RAW_DIR) -> Path:
    """Overwrite Users.csv with a PII-free copy (UserID, Age, Gender) before publishing."""
    path = _find_csv(Path(raw_dir), SCHEMAS["users"]["pattern"])
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    df.to_csv(path, index=False)
    return path
