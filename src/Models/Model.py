from typing import List, Optional

from huggingface_hub import (
    DatasetCardData,
    DatasetInfo,
    ModelCardData,
    ModelInfo,
)


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
        self.dataset_cards: dict[str, Optional[DatasetCardData]] = {}

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