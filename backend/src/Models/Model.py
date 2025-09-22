import os
from abc import ABC

from lib.Github_API_Manager import GitHubAPIManager
from lib.HuggingFace_API_Manager import HuggingFaceAPIManager


class Model(ABC):
    def __init__(self) -> None:
        github_token = os.getenv("GITHUB_TOKEN")
        self.github_manager = GitHubAPIManager(token=github_token)
        self.huggingface_manager = HuggingFaceAPIManager()
