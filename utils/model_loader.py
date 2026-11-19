from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "fake_job_model"
)


def load_model(
    model_path: str | Path = MODEL_DIR,
    *,
    device: str | torch.device | None = None,
) -> Tuple[object, object]:
    """Load the local tokenizer and sequence-classification model."""
    model_path = Path(model_path).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if device is not None:
        model = model.to(device)

    model.eval()
    return tokenizer, model
