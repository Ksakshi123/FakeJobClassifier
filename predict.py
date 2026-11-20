import joblib
import pandas as pd

from features import transform
from llm_check import analyze_posting, api_key

# Load the trained model bundle (fallback / secondary opinion)
bundle = joblib.load("Model/model_bundle.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
threshold_medium = bundle["threshold_medium"]

RISK_LEVELS = {
    "HIGH": "POTENTIALLY SUSPICIOUS",
    "MEDIUM": "UNCERTAIN — REVIEW RECOMMENDED",
    "LOW": "LOW RISK",
}
VERDICT_TO_LEVEL = {"FAKE": "HIGH", "UNCERTAIN": "MEDIUM", "REAL": "LOW"}


def risk_level(prob_fake):
    if prob_fake >= threshold:
        return "POTENTIALLY SUSPICIOUS"
    if prob_fake >= threshold_medium:
        return "UNCERTAIN — REVIEW RECOMMENDED"
    return "LOW RISK"


def predict_job(job_text):
    """Score a free-text job posting.

    Uses the two-model Groq LLM ensemble when GROQ_API_KEY is configured;
    otherwise falls back to the statistical model.
    Returns (risk_level, score_pct, reasons, engine).
    """
    llm = analyze_posting(job_text) if api_key() else None
    if llm is not None:
        reasons = list(llm.get("red_flags", []))
        if llm.get("summary"):
            reasons.append(llm["summary"])
        return (RISK_LEVELS[VERDICT_TO_LEVEL[llm["verdict"]]],
                llm["fraud_risk_score"], reasons, "llm-ensemble")

    row = {
        "title": "", "location": "", "department": "",
        "company_profile": "", "description": job_text,
        "requirements": "", "benefits": "",
        "telecommuting": "", "has_company_logo": "", "has_questions": "",
        "employment_type": "", "required_experience": "",
        "required_education": "", "industry": "", "function": "",
        "salary_range": "",
    }
    X = transform(pd.DataFrame([row]), bundle)
    prob_fake = float(model.predict_proba(X)[0, 1])
    return risk_level(prob_fake), round(prob_fake * 100, 1), [], "model"


if __name__ == "__main__":
    job = input("\nEnter a job description: ")
    level, score, reasons, engine = predict_job(job)
    print(f"\nRisk level: {level}")
    print(f"Suspicion score: {score}%  (engine: {engine})")
    for r in reasons:
        print(f"  - {r}")
    print("(Risk estimate only — verify the employer independently.)")
