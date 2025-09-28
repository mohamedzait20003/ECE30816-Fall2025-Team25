import os
import re
import json
from typing import List, Dict
from dotenv import load_dotenv
from Controllers.Controller import Controller
from Services.Metric_Model_Service import ModelMetricService

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


def classify_url(url: str) -> str:
    if re.match(r"^https?://huggingface\.co/datasets/", url):
        return "dataset"
    elif re.match(r"^https?://huggingface\.co/", url):
        return "model"
    elif re.match(r"^https?://github\.com/", url):
        return "code"
    else:
        return "unknown"


def parse_url_file(filepath: str) -> List[Dict[str, str]]:
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    category = classify_url(url)
                    results.append({"url": url, "type": category})
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return results


if __name__ == "__main__":
    fetcher = Controller()
    model_metric_service = ModelMetricService()

    dataset_links = [
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://huggingface.co/datasets/osunlp/UGround-V1-Data",
        "https://huggingface.co/datasets/xlangai/aguvis-stage2"
    ]
    code_link = "https://github.com/xlang-ai/OpenCUA"
    model_link = "https://huggingface.co/xlangai/OpenCUA-32B"

    print("Fetching model data...")
    model_data = fetcher.fetch(
        model_link,
        dataset_links=dataset_links,
        code_link=code_link
    )
    print("Model data fetched successfully!")

    print("Running model evaluation...")
    evaluation = model_metric_service.EvaluateModel(model_data)

    json_output = json.dumps(evaluation, separators=(',', ':'),
                             ensure_ascii=False)
    print(json_output.strip())
