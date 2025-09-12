import logging
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelInfo, ModelCardData
from typing import Optional


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

    if model_link.startswith("https://huggingface.co/"):
        return model_link[len("https://huggingface.co/"):]
    elif model_link.startswith("http://huggingface.co/"):
        return model_link[len("http://huggingface.co/"):]
    else:
        return model_link


load_dotenv()


class ModelData:
    ''' Data class to hold model information and metadata. '''

    def __init__(self, model_info: ModelInfo,) -> None:
        ''' Initializes ModelData with model information.
        Args:
            model_info (ModelInfo): The model information fetched from
                Hugging Face.
        '''
        self.model_info: ModelInfo = model_info

        self.model_card: Optional[ModelCardData] = \
            model_info.card_data

        self.readme: Optional[str] = None


class ModelFetcher:
    ''' Fetches model information from Hugging Face. '''

    def __init__(self) -> None:
        ''' Initializes the ModelFetcher with Hugging Face API. '''

        self.hf_api: HfApi = HfApi(
            endpoint="https://huggingface.co",
            token=get_hf_token(),
        )

    def fetch_model(self, model_link: str) -> ModelData:
        ''' Fetches model information from Hugging Face.
        Args:
            model_link (str): The link or ID of the model to fetch.
        Returns:
            ModelData: The fetched model data.
        '''

        model_id: str = model_link_to_id(model_link)
        model_info: ModelInfo = self.hf_api.model_info(model_id)
        return ModelData(model_info)


if __name__ == "__main__":
    fetcher = ModelFetcher()
    # model_link = "https://huggingface.co/gpt2"
    # model_link = "gpt2"
    model_link = "https://huggingface.co/prithivMLmods/Qwen-Image-HeadshotX"
    model_data = fetcher.fetch_model(model_link)
    print(model_data.model_card)
