import re


def clean_job_text(text: str) -> str:
    """Normalize whitespace and strip basic noise from job posting text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?\-$/]", " ", text)
    return text.strip()
