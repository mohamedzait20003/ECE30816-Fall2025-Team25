from abc import ABC

from lib.Github_API_Manager import GitHubAPIManager
from lib.HuggingFace_API_Manager import HuggingFaceAPIManager


class Model(ABC):
    def __init__(self) -> None:
        self.github_manager = GitHubAPIManager()
        self.huggingface_manager = HuggingFaceAPIManager()
