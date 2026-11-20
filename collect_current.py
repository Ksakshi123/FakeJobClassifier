"""Collect CURRENT (2025-2026) labeled job postings for validation/augmentation.

Sources (all free, no API key):
- Arbeitnow job board API  -> ~300 current legitimate postings
- Jobicy remote-jobs API   -> ~100 current legitimate remote postings
- Remotive API             -> current legitimate postings (small board)
- Dataset/current/scam_curated.csv (built by make_scam_curated.py) ->
  scam postings curated from verbatim news reports + documented scam patterns

Output: Dataset/current/legit_current.csv — legitimate current postings in the
same column schema as the Kaggle dataset, fraudulent=0.
"""

import json
import os
import re
import time
import urllib.request

import pandas as pd

OUT_DIR = "Dataset/current"
SCHEMA_COLS = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function",
    "fraudulent",
]


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def strip_html(html):
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def blank_row(**kw):
    row = {c: "" for c in SCHEMA_COLS}
    row.update(kw)
    row["fraudulent"] = 0
    return row


def fetch_arbeitnow():
    """https://www.arbeitnow.com/api/job-board-api — free job board aggregator."""
    rows = []
    try:
        data = get_json("https://www.arbeitnow.com/api/job-board-api")
    except Exception as e:
        print(f"  arbeitnow failed: {e}")
        return rows
    for j in data.get("data", []):
        desc = strip_html(j.get("description", ""))
        if len(desc) < 200:
            continue
        rows.append(blank_row(
            job_id=j.get("slug", ""),
            title=j.get("title", ""),
            location=j.get("location", "") or ("Remote" if j.get("remote") else ""),
            company_profile=j.get("company_name", ""),
            description=desc,
            telecommuting=1 if j.get("remote") else 0,
            employment_type=", ".join(j.get("job_types") or []) or "",
            industry=", ".join(j.get("tags") or [])[:200] or "",
            function=", ".join(j.get("tags") or [])[:200] or "",
        ))
    print(f"  arbeitnow: {len(rows)} jobs")
    return rows


def fetch_jobicy(count=100):
    """https://jobicy.com/api/v2/remote-jobs — free remote job API."""
    rows = []
    try:
        data = get_json(f"https://jobicy.com/api/v2/remote-jobs?count={count}")
    except Exception as e:
        print(f"  jobicy failed: {e}")
        return rows
    for j in data.get("jobs", []):
        desc = strip_html(j.get("jobDescription", ""))
        if len(desc) < 200:
            continue
        salary = ""
        if j.get("salaryMin") and j.get("salaryMax"):
            salary = f"{j.get('salaryCurrency','')}{j['salaryMin']}-{j['salaryMax']} {j.get('salaryPeriod','')}".strip()
        rows.append(blank_row(
            job_id=str(j.get("id", "")),
            title=j.get("jobTitle", ""),
            location=j.get("jobGeo", "") or "Remote",
            company_profile=j.get("companyName", ""),
            description=desc,
            salary_range=salary,
            telecommuting=1,
            employment_type=j.get("jobType", "") or "",
            industry=j.get("jobIndustry", "") or "",
            function=j.get("jobIndustry", "") or "",
        ))
    print(f"  jobicy: {len(rows)} jobs")
    return rows


def fetch_remotive():
    rows = []
    try:
        data = get_json("https://remotive.com/api/remote-jobs?limit=100")
    except Exception as e:
        print(f"  remotive failed: {e}")
        return rows
    for j in data.get("jobs", []):
        desc = strip_html(j.get("description", ""))
        if len(desc) < 200:
            continue
        rows.append(blank_row(
            job_id=str(j.get("id", "")),
            title=j.get("title", ""),
            location=j.get("candidate_required_location", "") or "Remote",
            company_profile=j.get("company_name", ""),
            description=desc,
            salary_range=j.get("salary", "") or "",
            telecommuting=1,
            has_company_logo=1 if j.get("company_logo") else 0,
            employment_type=j.get("job_type", "") or "",
            industry=j.get("category", "") or "",
            function=j.get("category", "") or "",
        ))
    print(f"  remotive: {len(rows)} jobs")
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    rows += fetch_arbeitnow()
    time.sleep(1)
    rows += fetch_jobicy()
    time.sleep(1)
    rows += fetch_remotive()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["title", "company_profile"])
    df.to_csv(f"{OUT_DIR}/legit_current.csv", index=False)
    print(f"\nSaved {len(df)} legit current postings "
          f"-> {OUT_DIR}/legit_current.csv")


if __name__ == "__main__":
    main()
