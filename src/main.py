import os
import re
import json
import sys
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


def extract_urls_from_line(line: str) -> List[str]:
    """Extract URLs from a comma-separated line, handling various formats."""
    urls = []
    # Handle different line formats
    # Remove all leading commas and whitespace from the entire line
    cleaned_line = line.lstrip(', \t')
    
    # Split by comma and clean up each URL
    parts = cleaned_line.split(',')
    for part in parts:
        url = part.strip()
        # Remove any remaining leading commas or whitespace
        while url.startswith(','):
            url = url[1:].strip()
        if url and url.startswith('http'):
            urls.append(url)
    return urls


def process_input_file(filepath: str) -> Dict[str, List[str]]:
    """Process input file and categorize URLs."""
    dataset_links = []
    model_links = []
    code_links = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                urls = extract_urls_from_line(line)
                for url in urls:
                    category = classify_url(url)
                    if category == "dataset":
                        dataset_links.append(url)
                    elif category == "model":
                        model_links.append(url)
                    elif category == "code":
                        code_links.append(url)
    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)
    
    return {
        "datasets": dataset_links,
        "models": model_links,
        "code": code_links
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    urls = process_input_file(input_file)

    fetcher = Controller()
    model_metric_service = ModelMetricService()

    if not urls["models"]:
        print("No model URLs found in input file")
        sys.exit(1)

    dataset_links = urls["datasets"] if urls["datasets"] else []
    code_link = urls["code"][0] if urls["code"] else None

    # Collect all model evaluations
    all_evaluations = []
    
    for model_link in urls["models"]:
        try:
            model_data = fetcher.fetch(
                model_link,
                dataset_links=dataset_links,
                code_link=code_link
            )

            evaluation = model_metric_service.EvaluateModel(model_data)
            all_evaluations.append(evaluation)

        except Exception as e:
            print(f"Error evaluating model {model_link}: {e}", file=sys.stderr)
            continue
    
    # Print all results once
    for evaluation in all_evaluations:
        json_output = json.dumps(evaluation, separators=(',', ':'),
                                 ensure_ascii=False)
        print(json_output.strip())
