import re
from typing import List, Dict, Optional

from Models.Model import Model
from Models.Manager_Models_Model import ModelManager
from Models.Manager_Dataset_Model import DatasetManager


class Controller:
    def __init__(self):
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()

    def fetch(self, main_link: str,
              dataset_links: Optional[List[str]] = None,
              code_link: Optional[str] = None) -> Optional[Model]:

        result_type = self.classify_url(main_link)

        if result_type == "model":
            return self.model_manager.where(
                main_link, dataset_links, code_link
            )
        elif result_type == "dataset":
            return self.dataset_manager.where(
                main_link, dataset_links, code_link
            )
        elif result_type == "code":
            if code_link is None:
                code_link = main_link
            return None
        else:
            return None

    def classify_url(self, url: str) -> str:
        if re.match(r"^https?://huggingface\.co/datasets/", url):
            return "dataset"
        elif re.match(r"^https?://huggingface\.co/", url):
            return "model"
        elif re.match(r"^https?://github\.com/", url):
            return "code"
        else:
            return "unknown"

    def parse_url_file(self, filepath: str) -> List[Dict[str, str]]:
        results = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        category = self.classify_url(url)
                        results.append({"url": url, "type": category})
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        return results
