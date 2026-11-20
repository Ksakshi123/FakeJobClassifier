# 🤖 AI Fake Job Detection System

A Machine Learning-based web application that analyzes job posting information and predicts whether a job posting is **FAKE** or **REAL**.
Available live on - https://fakejobclassifier-g963quarpuhabrimc3wcx5.streamlit.app/

## 🎯 Project Objective

The main objective of this project is to help job seekers identify potentially fraudulent job postings and reduce the risk of falling victim to fake recruitment scams.

The system uses Natural Language Processing (NLP) and Machine Learning techniques to analyze job posting content.

## 🧠 How It Works
<img width="1053" height="742" alt="image" src="https://github.com/user-attachments/assets/f51f8c6b-481f-45a7-9357-364055084578" />

1. The user enters job posting details.
2. The system combines the relevant text fields.
3. TF-IDF Vectorization converts the text into numerical features.
4. A Machine Learning classification model analyzes the features.
5. The system predicts whether the job is:
   - 🚨 FAKE JOB
   - ✅ REAL JOB
6. The application displays the prediction and confidence score.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- HTML
- CSS
- Git & GitHub

## 📊 Dataset

The project uses a job-posting dataset containing information such as:

- Job title
- Company profile
- Job description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Fraudulent label

The `fraudulent` field is used as the target variable.

## 🤖 Machine Learning

The system combines several feature groups:

- **Word TF-IDF** (1–2 grams) over all text fields
- **Character TF-IDF** (3–5 grams) for robustness to typos/obfuscation
- **Structured features**: company logo/questions flags, telecommuting,
  employment type, experience/education, industry/function (one-hot)
- **Engineered features**: field-presence flags and log text-lengths
  (scam postings often have missing company profiles, requirements, etc.)

- **Suspicious-pattern flags**: URLs, emails, phone numbers, messenger apps
  (WhatsApp/Telegram/etc.), currency amounts, urgency and guarantee language

Model selection is done by benchmark: `train_model.py` compares logistic
regression, multinomial/complement naive Bayes, SGD-log, and random forest
via out-of-fold PR-AUC and keeps the winner (logistic regression has won both
runs). A decision threshold tuned for fake-class F1 converts the probability
into a risk level.

## ⚠️ Important: this is a risk estimate, not a verdict

The UI reports **LOW RISK / UNCERTAIN / POTENTIALLY SUSPICIOUS** rather than
"FAKE" or "REAL". No classifier can guarantee a job is legitimate — always
verify employers independently (official websites, domain age, requests for
money/OTP/bank details are red flags regardless of the score).

## 📈 Model Performance

Metrics below are from a **held-out 20% split** the model never trained on.
Because only ~5% of postings are fraudulent, PR-AUC is more informative than
raw accuracy:

- ROC-AUC: **0.991**
- PR-AUC: **0.937**
- Fake-class at tuned threshold: precision **0.86**, recall **0.88**, F1 **0.87**

The training set includes partial-input augmentation (description-only
copies), so pasting just a job description doesn't trigger false alarms from
missing structured fields.

## 🧪 Validation on current (2025–2026) listings

The Kaggle training data is from ~2018, so we built an independent check:

- **398 legitimate postings** scraped from live job boards (Arbeitnow,
  Jobicy, Remotive APIs — `collect_current.py`)
- **14 verbatim scam messages** quoted in news/scam reports (Which?, PUNCH,
  ScamCheck, MEN, Free Press — `make_scam_curated.py`, eval-only)
- **19 synthetic scam postings** modeled on documented scam patterns
  (task scams, advance fees, impersonation — training augmentation only)

`prepare_current.py` splits these into `train_extra.csv` (augmentation) and
`eval_current.csv` (held-out). On the held-out current set:

- Before augmentation: scam recall **0.43** (6/14 caught), 0 false positives
- After augmentation: scam recall **0.93** (13/14), 0 false positives

Caveat: only 14 verbatim scam examples — treat this as indicative, not
proof. Re-run `python evaluate.py Dataset/current/eval_current.csv` anytime,
or point it at any CSV with the same columns + a `fraudulent` (0/1) label.

## 🤖 Optional LLM analysis (Groq)

If `GROQ_API_KEY` is set (environment variable or `.env` file — never commit
it), the app uses a Groq-hosted LLM (`llama-3.3-70b-versatile`) as the primary
detector. It reasons about scam signals semantically — impersonation, fee/OTP
requests, salary-vs-work mismatch — which catches sophisticated fakes that
bag-of-words models miss. The result shows the LLM's score, reasons, and the
statistical model's score as a secondary opinion. Without a key, the app
falls back to the statistical model automatically.

Get a free key at https://console.groq.com — then create `.env`:

```bash
GROQ_API_KEY=gsk_...
GROQ_API_KEY_2=gsk_...   # optional fallback — used if the first key fails/exhausts
```

## 🏢 Company & contact verification

`company_check.py` adds an independent signal alongside the model:
- Extracts domains/emails from the posting
- DNS-resolves them and looks up domain registration date via RDAP
- Flags domains registered <6 months ago, unresolvable domains, free-email-
  only contacts, and URL shorteners
- A "high" severity finding (e.g. domain <90 days old) escalates a LOW RISK
  result to UNCERTAIN — the model alone can't verify a company exists

## 🖥️ Web Application

The project includes a Flask-based web interface where users can enter job information and receive a prediction.

Example output:

**⚠️ POTENTIALLY SUSPICIOUS** (score 82.8%)

or

**🟢 LOW RISK** (score 9.5%)

or the middle band:

**🟡 UNCERTAIN — REVIEW RECOMMENDED**

## 📂 Project Structure

```text
AI-Fake-Job-Detection/
│
├── Dataset/
│   └── fake_job_postings.csv
│
├── Model/
│
├── static/
│
├── templates/
│   └── index.html
│
├── app.py
├── collect_current.py
├── company_check.py
├── evaluate.py
├── features.py
├── main.py
├── make_scam_curated.py
├── prepare_current.py
├── predict.py
├── test_model.py
├── train_model.py
├── dataset.csv
└── requirements.txt
