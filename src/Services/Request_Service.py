import requests
import logging
import os
import re

from huggingface_hub import DatasetInfo, HfApi, ModelInfo, hf_hub_download


class RequestService:
    GITHUB_API_BASE = "https://api.github.com"

    def __init__(self) -> None:
        """Initialize RequestService with Hugging Face API."""
        hf_token: str | None = os.getenv("HF_TOKEN")
        if hf_token is None:
            logging.error(
                "Hugging Face token not found in environment variables."
            )
            raise ValueError(
                "Hugging Face token not found in environment variables."
            )

        self.hf_api = HfApi(
            endpoint="https://huggingface.co",
            token=hf_token
        )

        self.hf_token = hf_token

    @staticmethod
    def model_link_to_id(model_link: str) -> str:
        ''' Converts a Hugging Face model link to a model ID. '''
        match = re.search(
            r"huggingface\.co/([^/]+/[^/]+)", model_link
        )
        if match:
            return match.group(1)
        raise ValueError(f"Invalid model link: {model_link}")

    @staticmethod
    def dataset_link_to_id(dataset_link: str) -> str:
        ''' Converts a Hugging Face dataset link to a dataset ID. '''
        match = re.search(
            r"huggingface\.co/datasets/([^/]+/[^/]+)", dataset_link
        )
        if match:
            return match.group(1)
        raise ValueError(f"Invalid dataset link: {dataset_link}")

    @staticmethod
    def code_link_to_repo(code_link: str) -> tuple[str, str]:
        ''' Converts a code repository link to a repo identifier. '''
        match = re.search(
            r"github\.com/([^/]+)/([^/]+)", code_link
        )
        if not match:
            raise ValueError(f"Invalid GitHub repo URL: {code_link}")
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo

    @staticmethod
    def github_request(
        path: str,
        token: str,
        params: dict | None = None
    ) -> dict | list:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        url = f"{RequestService.GITHUB_API_BASE}{path}"
        response = requests.get(
            url, headers=headers, params=params
        )
        if response.status_code != 200:
            raise ValueError(
                f"GitHub API request failed: {response.status_code} "
                f"{response.text}"
            )
        return response.json()

    @staticmethod
    def get_repo_contents(
        owner: str,
        repo: str,
        token: str,
        path: str = ""
    ) -> list:
        ''' Retrieves the contents of a GitHub repository. '''
        if path:
            path = f"/{path.lstrip('/')}"
        req = RequestService.github_request(
            path=f"/repos/{owner}/{repo}/contents{path}", token=token
        )
        assert isinstance(req, list), "Expected a list of contents"
        return req

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get model information from Hugging Face."""
        return self.hf_api.model_info(model_id)

    def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """Get dataset information from Hugging Face."""
        return self.hf_api.dataset_info(dataset_id)

    def download_model_readme(self, model_id: str) -> str | None:
        """Download model README file from Hugging Face."""
        try:
            return hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                token=self.hf_token,
            )
        except Exception as e:
            logging.warning(
                f"README.md not found for model {model_id}: {e}"
            )
            return None
