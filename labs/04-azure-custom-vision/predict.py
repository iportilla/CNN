"""Classify one image with a published Azure Custom Vision iteration.

Usage: python predict.py path/to/image.jpg
"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def prediction_url() -> str:
    endpoint = os.environ["CUSTOM_VISION_ENDPOINT"].rstrip("/")
    project = os.environ["CUSTOM_VISION_PROJECT_ID"]
    iteration = os.environ["CUSTOM_VISION_ITERATION"]
    return f"{endpoint}/customvision/v3.0/Prediction/{project}/classify/iterations/{iteration}/image"


def classify(image_path: str) -> list[dict]:
    headers = {
        "Prediction-Key": os.environ["CUSTOM_VISION_PREDICTION_KEY"],
        "Content-Type": "application/octet-stream",
    }
    with open(image_path, "rb") as image:
        response = requests.post(prediction_url(), headers=headers, data=image, timeout=30)
    response.raise_for_status()
    return response.json()["predictions"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for p in classify(sys.argv[1]):
        print(f"{p['tagName']:<12} {p['probability']:.1%}")
