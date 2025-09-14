import logging
import os
import re
from typing import List, Optional

import requests
from dotenv import load_dotenv
from huggingface_hub import (DatasetCard, DatasetInfo, HfApi, ModelCardData,
                             ModelInfo, hf_hub_download)

GITHUB_API_BASE = "https://api.github.com"

load_dotenv()


def get_hf_token() -> str:
    ''' Retrieves the Hugging Face token from environment variables. '''

    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    if hf_token is None:
        logging.error("Hugging Face token not found in environment variables.")
        raise ValueError(
            "Hugging Face token not found in environment variables."
        )
    return hf_token


def model_link_to_id(model_link: str) -> str:
    ''' Converts a Hugging Face model link to a model ID. '''

    match = re.search(r"huggingface\.co/([^/]+/[^/]+)", model_link)
    if match:
        return match.group(1)

    raise ValueError(f"Invalid model link: {model_link}")


def dataset_link_to_id(dataset_link: str) -> str:
    ''' Converts a Hugging Face dataset link to a dataset ID. '''

    match = re.search(r"huggingface\.co/datasets/([^/]+/[^/]+)", dataset_link)
    if match:
        return match.group(1)

    raise ValueError(f"Invalid dataset link: {dataset_link}")


def code_link_to_repo(code_link: str) -> tuple[str, str]:
    ''' Converts a code repository link to a repo identifier. '''

    match = re.search(r"github\.com/([^/]+)/([^/]+)", code_link)
    if not match:
        raise ValueError(f"Invalid GitHub repo URL: {code_link}")
    owner = match.group(1)
    repo = match.group(2).replace(".git", "")
    return owner, repo


def github_request(
    path: str,
    token: str,
    params: dict | None = None
) -> dict | list:
    ''''''
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"{GITHUB_API_BASE}{path}"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(
            (
                f"GitHub API request failed: {response.status_code} "
                f"{response.text}"
            )
        )
    return response.json()


def get_repo_contents(
    owner: str,
    repo: str,
    token: str,
    path: str = ""
) -> list:
    ''' Retrieves the contents of a GitHub repository. '''

    if path:
        path = f"/{path.lstrip('/')}"
    req = github_request(
        path=f"/repos/{owner}/{repo}/contents{path}",
        token=token
    )

    assert isinstance(req, list), "Expected a list of contents"
    return req


class ModelData:
    ''' Data class to hold model information and metadata. '''

    def __init__(self) -> None:
        ''' Initializes ModelData. '''

        self.id: str = ""
        self.info: Optional[ModelInfo] = None
        self.card: Optional[ModelCardData] = None
        self.readme_path: Optional[str] = None

        # Associated datasets
        self.dataset_ids: List[str] = []
        self.dataset_infos: dict[str, DatasetInfo] = {}
        self.dataset_cards: dict[str, DatasetCard] = {}

        # Code repository data
        self.repo_metadata: dict = {}
        self.repo_contents: list = []
        '''
        Only contains the contents of the root directory,
        if deeper traversal is needed, use get_repo_contents
        on desired subdirectory.
        '''
        self.repo_contributors: list = []
        self.repo_commit_history: list = []


class ModelFetcher:
    ''' Fetches model information from Hugging Face. '''

    def __init__(self) -> None:
        ''' Initializes the ModelFetcher with Hugging Face API. '''

        self.hf_api: HfApi = HfApi(
            endpoint="https://huggingface.co",
            token=get_hf_token(),
        )

    def fetch_model(self, model_link: str,
                    dataset_links: List[str] | None = None,
                    code_link: str | None = None) -> ModelData:
        ''' Fetches model information from Hugging Face.
        Args:
            model_link (str): The link or ID of the model to fetch.
            dataset_links (List[str] | None): Optional list of dataset links.
            code_link (str | None): Optional link to the code repository.
        Returns:
            ModelData: The fetched model data.
        '''

        model_data: ModelData = ModelData()

        model_data.id = model_link_to_id(model_link)
        model_data.info = self.hf_api.model_info(model_data.id)
        model_data.card = model_data.info.cardData
        try:
            model_data.readme_path = hf_hub_download(
                repo_id=model_data.id,
                filename="README.md",
                token=get_hf_token(),
            )
        except Exception as e:
            logging.warning(
                f"README.md not found for model {model_data.id}: {e}"
            )
            model_data.readme_path = None

        if dataset_links is not None:
            for dataset_link in dataset_links:
                try:
                    dataset_id = dataset_link_to_id(dataset_link)
                    model_data.dataset_ids.append(dataset_id)
                except ValueError as e:
                    logging.warning(f"Skipping invalid dataset link: {e}")

        for dataset_id in model_data.dataset_ids:
            try:
                dataset_info = self.hf_api.dataset_info(dataset_id)
                model_data.dataset_infos[dataset_id] = dataset_info
                model_data.dataset_cards[dataset_id] = dataset_info.cardData
            except Exception as e:
                logging.warning(f"Failed to fetch dataset {dataset_id}: {e}")

        owner, repo = None, None
        if code_link is not None:
            try:
                owner, repo = code_link_to_repo(code_link)
            except ValueError as e:
                logging.warning(f"Invalid code link provided: {e}")

        repo_metadata_result = github_request(
            path=f"/repos/{owner}/{repo}",
            token=os.getenv("GITHUB_TOKEN", "")
        ) if owner and repo else {}
        model_data.repo_metadata = (
            repo_metadata_result
            if isinstance(repo_metadata_result, dict)
            else {}
        )

        model_data.repo_contents = get_repo_contents(
            owner, repo, os.getenv("GITHUB_TOKEN", "")
        ) if owner and repo else []

        repo_contributors_result = github_request(
            path=f"/repos/{owner}/{repo}/contributors",
            token=os.getenv("GITHUB_TOKEN", "")
        ) if owner and repo else []
        model_data.repo_contributors = (
            repo_contributors_result if isinstance(repo_contributors_result,
                                                   list) else []
        )

        repo_commits_result = github_request(
            path=f"/repos/{owner}/{repo}/commits",
            token=os.getenv("GITHUB_TOKEN", ""),
            params={"per_page": 10}
        ) if owner and repo else []
        model_data.repo_commit_history = (
            repo_commits_result
            if isinstance(repo_commits_result, list)
            else []
        )

        return model_data


if __name__ == "__main__":

    fetcher = ModelFetcher()
    dataset_links = [
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://huggingface.co/datasets/osunlp/UGround-V1-Data",
        "https://huggingface.co/datasets/xlangai/aguvis-stage2"
    ]
    code_link = "https://github.com/xlang-ai/OpenCUA"
    model_link = "https://huggingface.co/xlangai/OpenCUA-32B"

    model_data = fetcher.fetch_model(model_link,
                                     dataset_links=dataset_links,
                                     code_link=code_link)

    print(f"Model ID: {model_data.id}")
    print("Model Card (first 100 characters):")
    if model_data.card is not None:
        print(model_data.card.__str__()[:100])
    else:
        print("No model card available.")

    print("README.md first 100 characters:")
    if model_data.readme_path is not None:
        with open(model_data.readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        print(readme_content[:100])
    else:
        print("README.md file not found for this model.")

    print(f"Associated Datasets: {model_data.dataset_ids}")
    print("Dataset Cards:")
    for dataset_id, dataset_card in model_data.dataset_cards.items():
        print(f"- {dataset_id} (first 100 characters):")
        if dataset_card is not None:
            print(dataset_card.__str__()[:100])
        else:
            print("No dataset card available.")

    print("Code Repository Metadata (first 5 items):")
    for key, value in list(model_data.repo_metadata.items())[:5]:
        print(f"- {key}: {value}")

    print("Code Repository Contents (first 5 items):")
    for item in model_data.repo_contents[:5]:
        print(f"- {item['name']} (type: {item['type']})")

    print("Code Repository Contributors (first 5):")
    for contributor in model_data.repo_contributors[:5]:
        print(
            f"- {contributor['login']} "
            f"(contributions: {contributor['contributions']})"
        )

    print("Code Repository Recent Commits (most recent 5):")
    for commit in model_data.repo_commit_history[:5]:
        commit_info = commit.get("commit", {})
        author_info = commit_info.get("author", {})
        print(
            f"- {commit_info.get('message', '').splitlines()[0]} "
            f"by {author_info.get('name', 'Unknown')} "
            f"on {author_info.get('date', 'Unknown')}"
        )
