
import os
import logging
from typing import List
from Models.Model import ModelData
from Services.Request_Service import RequestService


class ModelFetcher:
    ''' Fetches model information from Hugging Face. '''

    def __init__(self) -> None:
        '''Initializes the ModelFetcher with RequestService.'''
        self.service = RequestService()

    def fetch_model(self, model_link: str,
                    dataset_links: List[str] | None = None,
                    code_link: str | None = None) -> ModelData:
        ''' Fetches model information from Hugging Face. '''
        model_data: ModelData = ModelData()
        model_data.id = self.service.model_link_to_id(model_link)
        
        model_data.info = self.service.get_model_info(model_data.id)
        model_data.card = model_data.info.card_data
        
        model_data.readme_path = self.service.download_model_readme(
            model_data.id
        )

        if dataset_links is not None:
            for dataset_link in dataset_links:
                try:
                    dataset_id = self.service.dataset_link_to_id(dataset_link)
                    model_data.dataset_ids.append(dataset_id)
                except ValueError as e:
                    logging.warning(f"Skipping invalid dataset link: {e}")

        for dataset_id in model_data.dataset_ids:
            try:
                dataset_info = self.service.get_dataset_info(dataset_id)
                model_data.dataset_infos[dataset_id] = dataset_info
                # Use card_data instead of cardData for better type safety
                model_data.dataset_cards[dataset_id] = dataset_info.card_data
            except Exception as e:
                logging.warning(f"Failed to fetch dataset {dataset_id}: {e}")

        owner, repo = None, None
        if code_link is not None:
            try:
                owner, repo = self.service.code_link_to_repo(code_link)
            except ValueError as e:
                logging.warning(f"Invalid code link provided: {e}")

        repo_metadata_result = self.service.github_request(
            path=f"/repos/{owner}/{repo}",
            token=os.getenv("GITHUB_TOKEN", "")
        ) if owner and repo else {}
        model_data.repo_metadata = (
            repo_metadata_result
            if isinstance(repo_metadata_result, dict)
            else {}
        )

        model_data.repo_contents = self.service.get_repo_contents(
            owner, repo, os.getenv("GITHUB_TOKEN", "")
        ) if owner and repo else []

        repo_contributors_result = self.service.github_request(
            path=f"/repos/{owner}/{repo}/contributors",
            token=os.getenv("GITHUB_TOKEN", "")
        ) if owner and repo else []
        model_data.repo_contributors = (
            repo_contributors_result
            if isinstance(repo_contributors_result, list)
            else []
        )

        repo_commits_result = self.service.github_request(
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

    model_data = fetcher.fetch_model(
        model_link,
        dataset_links=dataset_links,
        code_link=code_link
    )

