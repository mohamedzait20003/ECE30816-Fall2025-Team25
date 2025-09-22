# URL Classifier 
import re
from typing import List, Dict


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
    urls = parse_url_file("urls.txt")
    for entry in urls:
        print(entry)