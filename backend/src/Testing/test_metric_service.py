"""
Unit tests for the Metric_Model_Service module.
Tests all metric evaluation functions.
"""
import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after adding to path
from Services.Metric_Model_Service import ModelMetricService  # noqa: E402
from lib.Metric_Result import MetricResult, MetricType  # noqa: E402
from Models.Model import Model  # noqa: E402


class TestModelMetricService:
    """Test cases for ModelMetricService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = ModelMetricService()

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_init(self, mock_llm_manager):
        """Test service initialization."""
        service = ModelMetricService()
        assert service.llm_manager is not None

    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        result = self.service.EvaluateModel("test model", "test dataset")
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.PERFORMANCE_CLAIMS
        assert result.value == 0.0
        assert "not yet implemented" in result.details["info"]

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_performance_claims_empty_model(self, mock_llm_manager):
        """Test performance claims evaluation with empty model."""
        mock_model = Mock(spec=Model)
        mock_model.readme_path = None
        mock_model.card = None
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.5"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluatePerformanceClaims(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.PERFORMANCE_CLAIMS

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    @patch('builtins.open')
    def test_evaluate_performance_claims_with_readme(self, mock_open,
                                                     mock_llm_manager):
        """Test performance claims evaluation with README."""
        mock_model = Mock(spec=Model)
        mock_model.readme_path = "/path/to/readme.md"
        mock_model.card = "Model card content"
        
        # Mock file reading
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "# Test Model\nAccuracy: 95%")
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.8"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluatePerformanceClaims(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.PERFORMANCE_CLAIMS

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_bus_factor(self, mock_llm_manager):
        """Test bus factor evaluation."""
        mock_model = Mock(spec=Model)
        mock_model.code_link = "https://github.com/test/repo"
        mock_model.code_description = "Test code description"
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.6"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluateBusFactor(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.BUS_FACTOR

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_size(self, mock_llm_manager):
        """Test size evaluation."""
        mock_model = Mock(spec=Model)
        mock_model.model_link = "https://huggingface.co/test/model"
        mock_model.model_description = "Test model"
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.7"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluateSize(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.SIZE_SCORE

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_ramp_up_time(self, mock_llm_manager):
        """Test ramp-up time evaluation."""
        mock_model = Mock(spec=Model)
        mock_model.readme_path = None
        mock_model.card = "Easy to use model"
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.8"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluateRampUpTime(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.RAMP_UP_TIME

    def test_evaluate_availability_no_links(self):
        """Test availability evaluation with no links."""
        mock_model = Mock(spec=Model)
        mock_model.dataset_links = []
        mock_model.code_link = None
        
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(
            mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.DATASET_AND_CODE_SCORE
        assert result.value == 0.0

    def test_evaluate_availability_partial_links(self):
        """Test availability evaluation with partial links."""
        mock_model = Mock(spec=Model)
        mock_model.dataset_links = ["https://huggingface.co/datasets/test"]
        mock_model.code_link = None
        
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(
            mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.DATASET_AND_CODE_SCORE
        assert 0.0 < result.value < 1.0

    def test_evaluate_availability_all_links(self):
        """Test availability evaluation with all links."""
        mock_model = Mock(spec=Model)
        mock_model.dataset_links = ["https://huggingface.co/datasets/test"]
        mock_model.code_link = "https://github.com/test/repo"
        
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(
            mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.DATASET_AND_CODE_SCORE
        assert result.value == 1.0

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_code_quality(self, mock_llm_manager):
        """Test code quality evaluation."""
        mock_model = Mock(spec=Model)
        mock_model.code_link = "https://github.com/test/repo"
        mock_model.code_description = "Well-documented code"
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.9"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluateCodeQuality(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.CODE_QUALITY

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_datasets_quality(self, mock_llm_manager):
        """Test datasets quality evaluation."""
        mock_model = Mock(spec=Model)
        mock_model.dataset_links = ["https://huggingface.co/datasets/test"]
        mock_model.dataset_descriptions = ["High quality dataset"]
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "0.85"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluateDatasetsQuality(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.DATASET_QUALITY

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_evaluate_license_with_readme(self, mock_llm_manager):
        """Test license evaluation with README."""
        mock_model = Mock(spec=Model)
        mock_model.readme_path = None
        mock_model.card = "MIT License"
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "1.0"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluateLicense(mock_model)
        
        assert isinstance(result, MetricResult)
        assert result.metric_type == MetricType.LICENSE

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_llm_parsing_invalid_response(self, mock_llm_manager):
        """Test handling of invalid LLM responses."""
        mock_model = Mock(spec=Model)
        mock_model.readme_path = None
        mock_model.card = "Test content"
        
        # Mock invalid LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "invalid_number"
        mock_llm_manager.return_value = mock_llm_instance
        
        service = ModelMetricService()
        result = service.EvaluatePerformanceClaims(mock_model)
        
        assert isinstance(result, MetricResult)
        # Should handle invalid response gracefully

    @patch('Services.Metric_Model_Service.PurdueLLMManager')
    def test_llm_parsing_boundary_values(self, mock_llm_manager):
        """Test handling of boundary LLM values."""
        mock_model = Mock(spec=Model)
        mock_model.readme_path = None
        mock_model.card = "Test content"
        
        # Test values outside 0-1 range
        test_values = ["-0.5", "1.5", "0.0", "1.0"]
        
        for test_value in test_values:
            mock_llm_instance = Mock()
            mock_llm_instance.generate_response.return_value = test_value
            mock_llm_manager.return_value = mock_llm_instance
            
            service = ModelMetricService()
            result = service.EvaluatePerformanceClaims(mock_model)
            
            assert isinstance(result, MetricResult)
            assert 0.0 <= result.value <= 1.0  # Should be clamped