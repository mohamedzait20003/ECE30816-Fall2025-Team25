from typing import List, Optional

from .Model import Model


class DatasetManager(Model):
    def __init__(self) -> None:
        super().__init__()
        self.id: str = ""
        self.title: str = ""
        self.description: str = ""
    
    def where(self, main_link: str,
              dataset_links: Optional[List[str]] = None,
              code_link: Optional[str] = None) -> 'DatasetManager':
        """
        Fetch dataset information from the provided link.
        
        Args:
            main_link: The main dataset URL
            dataset_links: Additional dataset links (optional)
            code_link: Associated code repository link (optional)
            
        Returns:
            DatasetManager: Instance with populated data
        """
        # TODO: Implement dataset fetching logic
        # This should use the huggingface_manager to fetch dataset info
        
        # For now, just store the link
        self.id = main_link
        
        return self
