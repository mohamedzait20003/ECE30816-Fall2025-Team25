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


def process_url_file(filepath: str) -> List[Dict[str, any]]:
    """
    Process URL file sequentially where dataset and code URLs
    are linked to the next model URL that appears.
    """
    model_configs = []
    current_datasets = []
    current_code = None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue

                category = classify_url(url)

                if category == "dataset":
                    current_datasets.append(url)
                elif category == "code":
                    current_code = url
                elif category == "model":
                    model_config = {
                        "model_url": url,
                        "dataset_urls": current_datasets[:],  # Copy the list
                        "code_url": current_code
                    }
                    model_configs.append(model_config)
                    
                    # Reset for next model (keep same dataset/code links
                    # unless new ones are specified)
                    # According to spec, links apply to the next model URL
                    current_datasets = []
                    current_code = None
    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)
    
    return model_configs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    model_configs = process_url_file(input_file)

    fetcher = Controller()
    model_metric_service = ModelMetricService()

    if not model_configs:
        print("No model URLs found in input file")
        sys.exit(1)

    # Process each model with its associated dataset and code links
    for config in model_configs:
        try:
            model_data = fetcher.fetch(
                config["model_url"],
                dataset_links=config["dataset_urls"],
                code_link=config["code_url"]
            )

            evaluation = model_metric_service.EvaluateModel(model_data)
            
            # Output immediately for each model (NDJSON format)
            json_output = json.dumps(evaluation, separators=(',', ':'),
                                     ensure_ascii=False)
            print(json_output.strip())

        except Exception as e:
            print(f"Error evaluating model {config['model_url']}: {e}",
                  file=sys.stderr)
            sys.exit(1)  # Exit with error code on failure
