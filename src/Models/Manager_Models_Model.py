import logging
import re
from huggingface_hub import (
    DatasetCardData,
    DatasetInfo,
    ModelCardData,
    ModelInfo,
)
from typing import List, Optional

from .Model import Model


class ModelManager(Model):
    def __init__(self) -> None:
        super().__init__()
        self.id: str = ""
        self.info: Optional[ModelInfo] = None
        self.card: Optional[ModelCardData] = None
        self.readme_path: Optional[str] = None

        self.dataset_ids: List[str] = []
        self.dataset_infos: dict[str, DatasetInfo] = {}
        self.dataset_cards: dict[str, Optional[DatasetCardData]] = {}

        self.repo_metadata: dict = {}
        self.repo_contents: list = []
        self.repo_contributors: list = []
        self.repo_commit_history: list = []

    def find_missing_links(self, model_link, dataset_link, code_link):
        discovered_datasets = []
        discovered_code = None

        try:
            model_id = self.huggingface_manager.model_link_to_id(
                model_link
            )

            model_info = self.huggingface_manager.get_model_info(model_id)
            if hasattr(model_info, 'cardData') and model_info.cardData:
                card_text = str(model_info.cardData)

                dataset_patterns = [
                    r'https://huggingface\.co/datasets/([^/\s]+/[^/\s]+)',
                    r'huggingface\.co/datasets/([^/\s]+/[^/\s]+)',
                    r'datasets/([^/\s\)]+/[^/\s\)]+)',
                ]

                for pattern in dataset_patterns:
                    matches = re.findall(pattern, card_text, re.IGNORECASE)
                    for match in matches:
                        if not match.startswith('http'):
                            dataset_url = (
                                f"https://huggingface.co/datasets/"
                                f"{match}"
                            )
                        else:
                            dataset_url = match
                        if dataset_url not in discovered_datasets:
                            discovered_datasets.append(dataset_url)

                code_patterns = [
                    r'https://github\.com/([^/\s\)]+/[^/\s\)]+)',
                    r'github\.com/([^/\s\)]+/[^/\s\)]+)',
                    r'\[.*?\]\(https://github\.com/([^/\s\)]+/[^/\s\)]+)\)',
                    r'repo:\s*([^/\s]+/[^/\s]+)',
                    r'code:\s*https://github\.com/([^/\s\)]+/[^/\s\)]+)',
                ]

                for pattern in code_patterns:
                    matches = re.findall(pattern, card_text, re.IGNORECASE)
                    if matches and not discovered_code:
                        match = matches[0]
                        match = match.rstrip('.,;)')

                        if not match.startswith('http'):
                            discovered_code = f"https://github.com/{match}"
                        else:
                            discovered_code = match
                        break

            if hasattr(model_info, 'tags') and model_info.tags:
                for tag in model_info.tags:
                    if 'dataset:' in tag:
                        dataset_name = tag.replace('dataset:', '').strip()
                        if '/' in dataset_name:
                            dataset_url = (
                                f"https://huggingface.co/datasets/"
                                f"{dataset_name}"
                            )
                            if dataset_url not in discovered_datasets:
                                discovered_datasets.append(dataset_url)

            if hasattr(model_info, 'modelId') and not discovered_code:
                model_id_parts = model_info.modelId.split('/')
                if len(model_id_parts) == 2:
                    org, model_name = model_id_parts
                    pattern = r'-(?:small|medium|large|xl|xxl|\d+[BMG]?)$'
                    base_name = re.sub(
                        pattern, '', model_name, flags=re.IGNORECASE
                    )

                    potential_repos = [
                        f"https://github.com/{org}/{base_name}",
                        f"https://github.com/{org}/{model_name}",
                        f"https://github.com/{org}/{model_name.lower()}",
                        f"https://github.com/{org.lower()}/{base_name}",
                    ]
                    discovered_code = potential_repos[0]

        except Exception as e:
            logging.warning(
                f"Could not fetch model info for {model_link}: {e}"
            )

        final_dataset_links = []
        if dataset_link and dataset_link.strip():
            final_dataset_links.append(dataset_link.strip())
        final_dataset_links.extend(discovered_datasets)

        final_code_link = code_link
        if not final_code_link or not final_code_link.strip():
            final_code_link = discovered_code

        return final_dataset_links, final_code_link

    def where(
        self,
        model_link: str,
        dataset_links: Optional[List[str]] = None,
        code_link: Optional[str] = None
    ) -> 'ModelManager':
        model_data = self

        model_data.id = self.huggingface_manager.model_link_to_id(model_link)
        model_data.info = self.huggingface_manager.get_model_info(
            model_data.id
        )
        model_data.card = (
            model_data.info.card_data if model_data.info else None
        )

        model_data.readme_path = (
            self.huggingface_manager.download_model_readme(model_data.id)
        )

        first_dataset_link = (
            dataset_links[0] if dataset_links and len(dataset_links) > 0
            else None
        )

        final_dataset_links, final_code_link = self.find_missing_links(
            model_link, first_dataset_link, code_link
        )

        if final_dataset_links:
            for dataset_link in final_dataset_links:
                try:
                    dataset_id = self.huggingface_manager.dataset_link_to_id(
                        dataset_link
                    )
                    if dataset_id not in model_data.dataset_ids:
                        model_data.dataset_ids.append(dataset_id)
                except ValueError as e:
                    logging.warning(f"Skipping invalid dataset link: {e}")

        for dataset_id in model_data.dataset_ids:
            try:
                dataset_info = self.huggingface_manager.get_dataset_info(
                    dataset_id
                )
                model_data.dataset_infos[dataset_id] = dataset_info
                model_data.dataset_cards[dataset_id] = dataset_info.card_data
            except Exception as e:
                logging.warning(f"Failed to fetch dataset {dataset_id}: {e}")

        # Use the final code link (provided or discovered)
        owner, repo = None, None
        if final_code_link is not None:
            try:
                owner, repo = self.github_manager.code_link_to_repo(
                    final_code_link
                )
            except ValueError as e:
                logging.warning(f"Invalid code link provided: {e}")

        self._fetch_github_data(model_data, owner, repo)

        return model_data

    def _fetch_github_data(self, model_data, owner: str, repo: str) -> None:
        if not (owner and repo):
            model_data.repo_metadata = {}
            model_data.repo_contents = []
            model_data.repo_contributors = []
            model_data.repo_commit_history = []
            return

        try:
            model_data.repo_metadata = (
                self.github_manager.get_repo_info(owner, repo)
            )
        except Exception as e:
            logging.warning(f"Failed to fetch repo metadata: {e}")
            model_data.repo_metadata = {}

        try:
            model_data.repo_contents = (
                self.github_manager.get_repo_contents(owner, repo)
            )
        except Exception as e:
            logging.warning(f"Failed to fetch repo contents: {e}")
            model_data.repo_contents = []

        try:
            repo_contributors_result = self.github_manager.github_request(
                path=f"/repos/{owner}/{repo}/contributors"
            )
            model_data.repo_contributors = (
                repo_contributors_result
                if isinstance(repo_contributors_result, list)
                else []
            )
        except Exception as e:
            logging.warning(f"Failed to fetch repo contributors: {e}")
            model_data.repo_contributors = []

        try:
            repo_commits_result = self.github_manager.github_request(
                path=f"/repos/{owner}/{repo}/commits",
                params={"per_page": 10}
            )
            model_data.repo_commit_history = (
                repo_commits_result
                if isinstance(repo_commits_result, list)
                else []
            )
        except Exception as e:
            logging.warning(f"Failed to fetch repo commits: {e}")
            model_data.repo_commit_history = []
