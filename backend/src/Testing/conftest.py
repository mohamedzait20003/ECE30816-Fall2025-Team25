import os
import sys
import pytest
from unittest.mock import Mock
from typing import Generator

from huggingface_hub import (
    ModelInfo, DatasetInfo, ModelCardData, DatasetCardData
)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Provide the path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def mock_env_vars() -> Generator[dict, None, None]:
    """Provide mock environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "HF_TOKEN": "test_hf_token_12345",
        "GITHUB_TOKEN": "test_github_token_67890"
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_model_data():
    """Provide sample model data for testing."""
    from Models.Model import ModelData
    
    model_data = ModelData()
    model_data.id = "test/sample-model"
    model_data.readme_path = "/path/to/readme.md"
    model_data.dataset_ids = ["test/dataset1", "test/dataset2"]
    model_data.repo_metadata = {
        "name": "sample-repo",
        "owner": {"login": "testuser"},
        "description": "A test repository"
    }
    model_data.repo_contents = [
        {"name": "README.md", "type": "file"},
        {"name": "src", "type": "dir"}
    ]
    model_data.repo_contributors = [
        {"login": "contributor1"},
        {"login": "contributor2"}
    ]
    model_data.repo_commit_history = [
        {
            "sha": "abc123",
            "commit": {
                "message": "Initial commit",
                "author": {"name": "Test Author"}
            }
        }
    ]
    
    return model_data


@pytest.fixture
def sample_urls():
    """Provide sample URLs for testing."""
    return {
        "model_link": "https://huggingface.co/test/sample-model",
        "dataset_links": [
            "https://huggingface.co/datasets/test/dataset1",
            "https://huggingface.co/datasets/test/dataset2"
        ],
        "code_link": "https://github.com/testuser/sample-repo"
    }


@pytest.fixture
def mock_huggingface_objects():
    """Provide mock Hugging Face API objects."""
    
    # Mock ModelInfo
    mock_model_info = Mock(spec=ModelInfo)
    mock_model_info.id = "test/sample-model"
    mock_model_info.card_data = Mock(spec=ModelCardData)
    
    # Mock DatasetInfo
    mock_dataset_info = Mock(spec=DatasetInfo)
    mock_dataset_info.id = "test/dataset1"
    mock_dataset_info.card_data = Mock(spec=DatasetCardData)
    
    return {
        "model_info": mock_model_info,
        "dataset_info": mock_dataset_info
    }


@pytest.fixture(autouse=True)
def clean_environment():
    """Automatically clean up environment variables after each test."""
    yield
    # Clean up any test-specific environment variables
    test_vars = ["TEST_HF_TOKEN", "TEST_GITHUB_TOKEN"]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "convergence: mark test as convergence test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "test_model_fetcher" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        if "convergence" in item.nodeid:
            item.add_marker(pytest.mark.convergence)
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker for tests that might make real API calls
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup hook that runs before each test."""
    # Skip integration tests if running in CI without proper tokens
    if "integration" in [mark.name for mark in item.iter_markers()]:
        if not os.getenv("HF_TOKEN") or not os.getenv("GITHUB_TOKEN"):
            pytest.skip("Integration tests require HF_TOKEN and GITHUB_TOKEN")


# Custom pytest fixtures for specific test scenarios
@pytest.fixture
def temporary_test_files(tmp_path):
    """Create temporary test files for testing file operations."""
    test_files = {}
    
    # Create a temporary README file
    readme_file = tmp_path / "README.md"
    readme_file.write_text("# Test Model\nThis is a test model.")
    test_files["readme"] = str(readme_file)
    
    # Create a temporary requirements file
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("torch>=1.9.0\ntransformers>=4.20.0")
    test_files["requirements"] = str(requirements_file)
    
    return test_files