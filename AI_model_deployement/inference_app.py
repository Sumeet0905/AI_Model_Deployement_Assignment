"""
inference_app.py

This script demonstrates a basic AI model deployment workflow:
- Downloads a pre-trained model file using requests
- Loads a tokenizer and model using Hugging Face Transformers
- Performs sentiment analysis using PyTorch
- Supports optional command-line input
"""

import os
import sys
import requests
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Model configuration
MODEL_URL = "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/pytorch_model.bin"
MODEL_PATH = "pytorch_model.bin"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def download_model():
    """Download the model file if it does not already exist."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")


def load_model():
    """Load the tokenizer and model from Hugging Face."""
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_sentiment(text, tokenizer, model):
    """Predict sentiment (Positive / Negative) for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "Positive" if prediction == 1 else "Negative"


def main():
    download_model()
    tokenizer, model = load_model()

    # Use command-line input if provided, otherwise use default examples
    if len(sys.argv) > 1:
        sentences = [" ".join(sys.argv[1:])]
    else:
        sentences = [
            "I love this product!",
            "This movie was terrible."
        ]

    print("\nSentiment Analysis Results:\n")
    for sentence in sentences:
        sentiment = predict_sentiment(sentence, tokenizer, model)
        print(f"Text: {sentence}")
        print(f"Prediction: {sentiment}\n")


if __name__ == "__main__":
    main()