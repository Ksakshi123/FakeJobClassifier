# AI Fake Job Classification

A Streamlit application that uses a fine-tuned DistilBERT model to identify fraudulent job postings and explains predictions with SHAP.

## Project Structure

```text
FakeJobClassification/
├── app.py
├── requirements.txt
├── README.md
├── pages/
│   ├── 1_Predict.py
│   ├── 2_Model_Performance.py
│   └── 3_About.py
├── models/
│   ├── fake_job_model/
│   └── vectorizer.pkl (optional)
└── utils/
    └── preprocessing.py
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the fine-tuned Hugging Face model files in `models/fake_job_model/`.

## Run the App

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. The prediction page accepts a job description and reports the real/fake probabilities and influential words.

## Model

- Architecture: DistilBERT
- Maximum sequence length: 256 tokens
- Explainability: SHAP text masking
