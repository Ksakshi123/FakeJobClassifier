"""Minimal Streamlit UI for the job-fraud detector.

Run:  streamlit run streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st

from features import transform
from llm_check import analyze_posting, api_key
from predict import predict_job  # ML fallback

bundle = joblib.load("Model/model_bundle.pkl")


def _ml_score(text):
    """Statistical model suspicion % for a description-only posting."""
    row = {c: "" for c in [
        "title", "location", "department", "company_profile", "requirements",
        "benefits", "telecommuting", "has_company_logo", "has_questions",
        "employment_type", "required_experience", "required_education",
        "industry", "function", "salary_range"]}
    row["description"] = text
    X = transform(pd.DataFrame([row]), bundle)
    return round(float(bundle["model"].predict_proba(X)[0, 1]) * 100, 1)

st.set_page_config(page_title="Fraudulent Job Detection", page_icon="🛡️", layout="centered")

VERDICT_STYLE = {
    "FAKE":      ("POTENTIALLY SUSPICIOUS", "#dc2626", "#fef2f2", "#991b1b"),
    "UNCERTAIN": ("UNCERTAIN — REVIEW",     "#d97706", "#fffbeb", "#92400e"),
    "REAL":      ("LOW RISK",                "#059669", "#ecfdf5", "#065f46"),
}

st.markdown("""
<style>
    .block-container { padding-top: 3rem; max-width: 760px; }
    .verdict-card {
        border-radius: 14px; padding: 22px 26px; margin: 18px 0;
        border-left: 6px solid;
    }
    .verdict-title { font-size: 1.5rem; font-weight: 700; }
    .metric-row { display: flex; gap: 28px; margin-top: 10px; }
    .metric-val { font-size: 1.6rem; font-weight: 700; }
    .metric-lbl { font-size: 0.8rem; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.05em; }
    .flag { margin: 4px 0; }
    .summary { margin-top: 12px; font-size: 0.95rem; color: #334155; }
    .foot { margin-top: 26px; font-size: 0.8rem; color: #94a3b8; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Fraudulent Job Detection")
st.caption("Paste a job posting — get an instant risk assessment.")

jd = st.text_area(
    "Job description",
    height=280,
    placeholder="Paste the full job posting here — title, company, responsibilities, requirements, contact details, salary…",
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    run = st.button("Analyze", type="primary", use_container_width=True)
with col_hint:
    if not api_key():
        st.caption("⚠️ No `GROQ_API_KEY` in `.env` — using statistical model")

if run and jd.strip():
    with st.spinner("Analyzing with two AI models…"):
        llm = analyze_posting(jd) if api_key() else None

    if llm is not None:
        title, accent, bg, fg = VERDICT_STYLE[llm["verdict"]]
        st.markdown(f"""
        <div class="verdict-card" style="background:{bg}; border-color:{accent};">
            <div class="verdict-title" style="color:{fg};">{title}</div>
            <div class="metric-row">
                <div><div class="metric-val" style="color:{accent};">{llm['fraud_risk_score']}%</div>
                     <div class="metric-lbl">fraud risk</div></div>
                <div><div class="metric-val" style="color:{accent};">{llm['confidence']}%</div>
                     <div class="metric-lbl">confidence</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        if llm.get("red_flags"):
            st.markdown("**Red flags detected**")
            for f in llm["red_flags"]:
                st.markdown(f'<div class="flag">🚩 {f}</div>', unsafe_allow_html=True)

        if llm.get("summary"):
            st.markdown(f'<div class="summary">{llm["summary"]}</div>',
                        unsafe_allow_html=True)

        with st.expander("Criterion breakdown"):
            for name, crit in (llm.get("criteria") or {}).items():
                if isinstance(crit, dict):
                    st.markdown(
                        f"**{name.replace('_', ' ').title()}** — "
                        f"`{crit.get('status', '?')}` ({crit.get('risk', '?')}/100)\n\n"
                        f"{crit.get('reason', '')}"
                    )

        # statistical model as secondary opinion
        try:
            st.caption(f"Statistical model score: {_ml_score(jd)}%")
        except Exception:
            pass

    else:
        # ML fallback
        level, score, reasons, engine = predict_job(jd)
        css = {"POTENTIALLY SUSPICIOUS": "fake", "UNCERTAIN — REVIEW RECOMMENDED": "medium"}.get(level, "real")
        color = {"fake": "#dc2626", "medium": "#d97706", "real": "#059669"}[css]
        st.markdown(f"""
        <div class="verdict-card" style="background:#f8fafc; border-color:{color};">
            <div class="verdict-title" style="color:{color};">{level}</div>
            <div class="metric-row">
                <div><div class="metric-val" style="color:{color};">{score}%</div>
                     <div class="metric-lbl">suspicion score</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.caption("AI analysis unavailable — statistical model result shown.")

st.markdown(
    '<div class="foot">Indicative risk assessment only — always verify the employer independently.</div>',
    unsafe_allow_html=True,
)
