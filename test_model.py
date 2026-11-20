import joblib
import pandas as pd

from features import transform

# Load the trained model bundle
bundle = joblib.load("Model/model_bundle.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
threshold_medium = bundle["threshold_medium"]


def risk_level(prob_fake):
    if prob_fake >= threshold:
        return "POTENTIALLY SUSPICIOUS"
    if prob_fake >= threshold_medium:
        return "UNCERTAIN"
    return "LOW RISK"


# Test job postings
BLANK = {c: "" for c in [
    "title", "location", "department", "company_profile", "requirements",
    "benefits", "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "salary_range",
]}


def score_text(text, **overrides):
    row = {**BLANK, "description": text, **{k: str(v) for k, v in overrides.items()}}
    X = transform(pd.DataFrame([row]), bundle)
    p = float(model.predict_proba(X)[0, 1])
    return risk_level(p), round(p * 100, 1)


test_jobs = [

    # FAKE JOB 1
    (
        """
        Work from home and earn 50000 per week.
        No experience required.
        Pay a registration fee of 2000 to get started.
        Send your bank details and OTP immediately.
        Limited vacancies. Apply now.
        """,
        {},
    ),

    # REAL JOB 1
    (
        """
        Data Analyst required at a technology company.
        Responsibilities include analyzing datasets, creating reports,
        preparing dashboards and working with business teams.
        Candidates should have knowledge of Python, SQL, Excel and statistics.
        Bachelor's degree in Data Science or a related field preferred.
        Full-time position with a formal interview process.
        """,
        {},
    ),

    # FAKE JOB 2
    (
        """
        Congratulations! You have been selected for a high-paying online job.
        Earn 100000 per month without any interview or experience.
        Pay a small security deposit to receive your employee ID.
        Send your bank account information to complete registration.
        """,
        {},
    ),

    # REAL JOB 2
    (
        """
        Software Developer Intern.
        The intern will assist the development team with software testing,
        debugging and documentation.
        Knowledge of Python or Java is preferred.
        Candidates will participate in a technical interview.
        This is a three-month paid internship.
        """,
        {},
    ),

    # FAKE JOB 3
    (
        """
        Make money instantly from home.
        We guarantee huge weekly income with zero skills required.
        No interview is necessary.
        Pay a joining fee today and start earning immediately.
        Contact us through the provided messaging number.
        """,
        {},
    ),
]


print("\n====================================")
print("AI FAKE JOB DETECTION - TESTING")
print("====================================\n")


for number, (job, extra) in enumerate(test_jobs, start=1):

    level, score = score_text(job, **extra)

    print(f"Test {number}: {level}")
    print(f"Suspicion score: {score}%")
    print("------------------------------------")


print("\nTesting completed successfully!")
