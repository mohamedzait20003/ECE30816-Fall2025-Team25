import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing other modules
load_dotenv()

from .Controllers.ModelFetcher import ModelFetcher

if __name__ == "__main__":
    
    fetcher = ModelFetcher()
    dataset_links = [
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://huggingface.co/datasets/osunlp/UGround-V1-Data",
        "https://huggingface.co/datasets/xlangai/aguvis-stage2"
    ]
    
    code_link = "https://github.com/xlang-ai/OpenCUA"
    model_link = "https://huggingface.co/xlangai/OpenCUA-32B"

    model_data = fetcher.fetch_model(model_link, dataset_links=dataset_links, code_link=code_link)

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
