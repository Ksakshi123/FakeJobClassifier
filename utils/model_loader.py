
from typing import Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Hugging Face model repository
MODEL_ID = "ksakshi123/fake-job-distilbert"


def load_model(
    model_path: str = MODEL_ID,
    *,
    device: str | torch.device | None = None,
) -> Tuple[object, object]:
    """
    Load tokenizer and fine-tuned classification model
    from Hugging Face or a local directory.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    if device is not None:
        model = model.to(device)

    model.eval()

    return tokenizer, model
