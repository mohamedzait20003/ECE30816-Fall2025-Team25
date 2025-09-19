import os
import sys
import pytest
from unittest.mock import Mock, patch
from huggingface_hub import (
    ModelInfo, DatasetInfo, ModelCardData, DatasetCardData
)


from Controllers.ModelFetcher import ModelFetcher
from Models.Model import ModelData
from Services.Request_Service import RequestService

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestModelFetcher:
    """Test suite for ModelFetcher class."""

    @pytest.fixture
    def mock_request_service(self):
        """Create a mock RequestService for testing."""
        mock_service = Mock(spec=RequestService)
        return mock_service

    @pytest.fixture
    def model_fetcher(self, mock_request_service):
        """Create a ModelFetcher instance with mocked RequestService."""
        with patch('Controllers.ModelFetcher.RequestService',
                   return_value=mock_request_service):
            fetcher = ModelFetcher()
            fetcher.service = mock_request_service
            return fetcher

    @pytest.fixture
    def sample_model_info(self):
        """Create a sample ModelInfo object for testing."""
        mock_model_info = Mock(spec=ModelInfo)
        mock_model_info.card_data = Mock(spec=ModelCardData)
        return mock_model_info

    @pytest.fixture
    def sample_dataset_info(self):
        """Create a sample DatasetInfo object for testing."""
        mock_dataset_info = Mock(spec=DatasetInfo)
        mock_dataset_info.card_data = Mock(spec=DatasetCardData)
        return mock_dataset_info

    def test_init(self):
        """Test ModelFetcher initialization."""
        with patch('Controllers.ModelFetcher.RequestService') as \
                mock_service_class:
            mock_service_instance = Mock()
            mock_service_class.return_value = mock_service_instance
            
            fetcher = ModelFetcher()
            
            assert fetcher.service == mock_service_instance
            mock_service_class.assert_called_once()

    def test_fetch_model_basic(self, model_fetcher, mock_request_service,
                               sample_model_info):
        """Test basic model fetching without datasets or code link."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        readme_path = "/path/to/readme.md"

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        mock_request_service.download_model_readme.return_value = readme_path

        # Act
        result = model_fetcher.fetch_model(model_link)

        # Assert
        assert isinstance(result, ModelData)
        assert result.id == model_id
        assert result.info == sample_model_info
        assert result.card == sample_model_info.card_data
        assert result.readme_path == readme_path
        
        mock_request_service.model_link_to_id.assert_called_once_with(
            model_link
        )
        mock_request_service.get_model_info.assert_called_once_with(model_id)
        mock_request_service.download_model_readme.assert_called_once_with(
            model_id
        )

    def test_fetch_model_with_datasets(self, model_fetcher,
                                       mock_request_service,
                                       sample_model_info,
                                       sample_dataset_info):
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        dataset_links = [
            "https://huggingface.co/datasets/test/dataset1",
            "https://huggingface.co/datasets/test/dataset2"
        ]

        dataset_ids = ["test/dataset1", "test/dataset2"]

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        mock_request_service.download_model_readme.return_value = (
            "/path/to/readme.md"
        )
        mock_request_service.dataset_link_to_id.side_effect = dataset_ids
        mock_request_service.get_dataset_info.return_value = (
            sample_dataset_info
        )

        result = model_fetcher.fetch_model(
            model_link, dataset_links=dataset_links
        )

        assert len(result.dataset_ids) == 2
        assert result.dataset_ids == dataset_ids
        assert len(result.dataset_infos) == 2
        assert len(result.dataset_cards) == 2
        
        for dataset_id in dataset_ids:
            assert result.dataset_infos[dataset_id] == sample_dataset_info
            assert (result.dataset_cards[dataset_id] ==
                    sample_dataset_info.card_data)

    def test_fetch_model_with_invalid_dataset(self, model_fetcher,
                                              mock_request_service,
                                              sample_model_info):

        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        dataset_links = ["invalid_dataset_link"]

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        mock_request_service.download_model_readme.return_value = (
            "/path/to/readme.md"
        )
        mock_request_service.dataset_link_to_id.side_effect = ValueError(
            "Invalid dataset link"
        )

        with patch('Controllers.ModelFetcher.logging.warning') as mock_warning:
            result = model_fetcher.fetch_model(
                model_link, dataset_links=dataset_links
            )

        assert len(result.dataset_ids) == 0
        mock_warning.assert_called_once()

    def test_fetch_model_with_code_link(self, model_fetcher,
                                        mock_request_service,
                                        sample_model_info):
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        code_link = "https://github.com/owner/repo"
        owner, repo = "owner", "repo"
        
        mock_repo_metadata = {"name": "repo", "owner": {"login": "owner"}}
        mock_repo_contents = [{"name": "file.py", "type": "file"}]
        mock_contributors = [{"login": "user1"}]
        mock_commits = [
            {"sha": "abc123", "commit": {"message": "Initial commit"}}
        ]

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        mock_request_service.download_model_readme.return_value = (
            "/path/to/readme.md"
        )
        mock_request_service.code_link_to_repo.return_value = (owner, repo)
        mock_request_service.get_repo_contents.return_value = (
            mock_repo_contents
        )

        def github_request_side_effect(path, token, params=None):
            if "/repos/owner/repo$" in path:
                return mock_repo_metadata
            elif "/contributors" in path:
                return mock_contributors
            elif "/commits" in path:
                return mock_commits
            return {}

        (mock_request_service.github_request.side_effect
         ) = github_request_side_effect

        # Act
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'test_token'}):
            result = model_fetcher.fetch_model(model_link, code_link=code_link)

        # Assert
        assert result.repo_metadata == mock_repo_metadata
        assert result.repo_contents == mock_repo_contents
        assert result.repo_contributors == mock_contributors
        assert result.repo_commit_history == mock_commits

    def test_fetch_model_with_invalid_code_link(self, model_fetcher,
                                                mock_request_service,
                                                sample_model_info):
        """Test model fetching with invalid GitHub code link."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        code_link = "invalid_code_link"

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        mock_request_service.download_model_readme.return_value = (
            "/path/to/readme.md"
        )
        mock_request_service.code_link_to_repo.side_effect = ValueError(
            "Invalid code link"
        )

        # Act
        with patch('Controllers.ModelFetcher.logging.warning') as mock_warning:
            result = model_fetcher.fetch_model(model_link, code_link=code_link)

        # Assert
        assert result.repo_metadata == {}
        assert result.repo_contents == []
        assert result.repo_contributors == []
        assert result.repo_commit_history == []
        mock_warning.assert_called_once()

    def test_fetch_model_github_api_failure(self, model_fetcher,
                                            mock_request_service,
                                            sample_model_info):
        """Test model fetching when GitHub API requests fail."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        code_link = "https://github.com/owner/repo"
        owner, repo = "owner", "repo"

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        (mock_request_service.download_model_readme.return_value
         ) = "/path/to/readme.md"
        mock_request_service.code_link_to_repo.return_value = (owner, repo)
        mock_request_service.github_request.return_value = "not_a_dict_or_list"
        mock_request_service.get_repo_contents.return_value = []

        # Act
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'test_token'}):
            result = model_fetcher.fetch_model(model_link, code_link=code_link)

        # Assert
        assert result.repo_metadata == {}
        assert result.repo_contributors == []
        assert result.repo_commit_history == []

    def test_fetch_model_dataset_info_failure(self, model_fetcher,
                                              mock_request_service,
                                              sample_model_info):
        """Test model fetching when dataset info retrieval fails."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        dataset_links = ["https://huggingface.co/datasets/test/dataset1"]
        dataset_id = "test/dataset1"

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        (mock_request_service.download_model_readme.return_value
         ) = "/path/to/readme.md"
        mock_request_service.dataset_link_to_id.return_value = dataset_id
        (mock_request_service.get_dataset_info.side_effect
         ) = Exception("Dataset fetch failed")

        # Act
        with patch('Controllers.ModelFetcher.logging.warning') as mock_warning:
            result = model_fetcher.fetch_model(
                model_link, dataset_links=dataset_links
            )

        # Assert
        assert len(result.dataset_ids) == 1
        assert result.dataset_ids[0] == dataset_id
        assert len(result.dataset_infos) == 0  # Should be empty due to failure
        assert len(result.dataset_cards) == 0  # Should be empty due to failure
        mock_warning.assert_called()

    def test_fetch_model_no_github_token(self, model_fetcher,
                                         mock_request_service,
                                         sample_model_info):
        """Test model fetching without GitHub token in environment."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        code_link = "https://github.com/owner/repo"

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        (mock_request_service.download_model_readme.return_value
         ) = "/path/to/readme.md"
        mock_request_service.code_link_to_repo.return_value = ("owner", "repo")

        # Act
        with patch.dict(os.environ, {},
                        clear=True):  # Clear environment variables
            result = model_fetcher.fetch_model(model_link, code_link=code_link)

        # Assert
        assert result.repo_metadata == {}
        assert result.repo_contents == []
        assert result.repo_contributors == []
        assert result.repo_commit_history == []

    @pytest.mark.parametrize("model_link,expected_calls", [
        ("https://huggingface.co/test/model", 1),
        ("https://huggingface.co/another/model", 1),
    ])
    def test_fetch_model_parametrized(self, model_fetcher,
                                      mock_request_service,
                                      sample_model_info, model_link,
                                      expected_calls):
        """Parametrized test for different model links."""
        # Arrange
        mock_request_service.model_link_to_id.return_value = "test/model"
        mock_request_service.get_model_info.return_value = sample_model_info
        (mock_request_service.download_model_readme.return_value
         ) = "/path/to/readme.md"

        # Act
        result = model_fetcher.fetch_model(model_link)

        # Assert
        assert isinstance(result, ModelData)
        assert mock_request_service.model_link_to_id.call_count == \
            expected_calls

    def test_fetch_model_integration_like(self):
        """Integration-like test that doesn't mock RequestService
        constructor."""
        # This test would require actual network calls or more
        # sophisticated mocking
        # For now, it's a placeholder for future integration testing
        pass

    def test_fetch_model_error_handling(self, model_fetcher,
                                        mock_request_service):
        """Test error handling in fetch_model method."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        (mock_request_service.model_link_to_id.side_effect
         ) = ValueError("Invalid model link")

        # Act & Assert
        with pytest.raises(ValueError):
            model_fetcher.fetch_model(model_link)

    def test_fetch_model_empty_datasets_list(self, model_fetcher,
                                             mock_request_service,
                                             sample_model_info):
        """Test model fetching with empty dataset list."""
        # Arrange
        model_link = "https://huggingface.co/test/model"
        model_id = "test/model"
        dataset_links = []

        mock_request_service.model_link_to_id.return_value = model_id
        mock_request_service.get_model_info.return_value = sample_model_info
        (mock_request_service.download_model_readme.return_value
         ) = "/path/to/readme.md"

        # Act
        result = model_fetcher.fetch_model(
            model_link, dataset_links=dataset_links
        )

        # Assert
        assert len(result.dataset_ids) == 0
        assert len(result.dataset_infos) == 0
        assert len(result.dataset_cards) == 0
        mock_request_service.dataset_link_to_id.assert_not_called()


class TestModelFetcherEdgeCases:
    """Additional edge case tests for ModelFetcher."""

    def test_fetch_model_none_readme_path(self):
        """Test when README download returns None."""
        # This would test the case where download_model_readme returns None
        pass

    def test_fetch_model_malformed_github_response(self):
        """Test handling of malformed GitHub API responses."""
        # This would test various malformed responses from GitHub API
        pass

    def test_fetch_model_network_timeout(self):
        """Test handling of network timeouts."""
        # This would test timeout scenarios
        pass


if __name__ == "__main__":
    pytest.main([__file__])

