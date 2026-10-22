import pandas as pd
import pytest

from edupro.data import build_fact


@pytest.fixture()
def tables():
    teachers = pd.DataFrame({
        "TeacherID": ["T1", "T2", "T3"], "TeacherName": ["A", "B", "C"], "Age": [30, 40, 50],
        "Gender": ["Female", "Male", "Female"], "Expertise": ["Finance", "Design", "Finance"],
        "YearsOfExperience": [2, 8, 15], "TeacherRating": [2.0, 3.0, 4.5]})
    courses = pd.DataFrame({
        "CourseID": ["C1", "C2"], "CourseName": ["x", "y"], "CourseCategory": ["Finance", "Design"],
        "CourseType": ["Paid", "Free"], "CourseLevel": ["Beginner", "Advanced"],
        "CoursePrice": [100.0, 0.0], "CourseDuration": [10.0, 20.0], "CourseRating": [4.0, 2.0]})
    users = pd.DataFrame({"UserID": ["U1", "U2"], "Age": [20, 30], "Gender": ["Male", "Female"]})
    tx = pd.DataFrame({
        "TransactionID": [f"X{i}" for i in range(6)],
        "UserID": ["U1", "U2", "U1", "U2", "U1", "U2"],
        "CourseID": ["C1", "C1", "C2", "C2", "C1", "C2"],
        "TransactionDate": ["1/1/2025", "15/2/2025", "3/3/2025", "4/4/2025", "5/5/2025", "6/6/2025"],
        "Amount": [100.0, 100.0, 0.0, 0.0, 100.0, 0.0],
        "PaymentMethod": ["PayPal"] * 6,
        "TeacherID": ["T1", "T2", "T3", "T3", "T3", "T1"]})
    return {"teachers": teachers, "courses": courses, "transactions": tx, "users": users}


@pytest.fixture()
def fact(tables):
    return build_fact(tables)
