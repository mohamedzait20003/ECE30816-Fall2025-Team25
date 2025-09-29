"""
Unit tests for the main.py module functionality.
Tests input parsing, evaluation timing, and output formatting.
"""
import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after adding to path
from main import (  # noqa: E402
    parse_input,
    time_evaluation,
    extract_model_name,
    format_size_score,
    run_evaluations_sequential,
    run_evaluations_parallel,
    find_missing_links,
    print_timing_summary
)
from lib.Metric_Result import MetricResult, MetricType  # noqa: E402


class TestParseInput:
    """Test cases for input file parsing."""

    def test_parse_empty_file(self):
        """Test parsing an empty input file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write("")
            temp_path = f.name

        try:
            result = parse_input(temp_path)
            assert result == []
        finally:
            os.unlink(temp_path)

    def test_parse_single_model_link(self):
        """Test parsing file with single model link."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write("https://huggingface.co/test/model")
            temp_path = f.name

        try:
            result = parse_input(temp_path)
            expected = [{
                'model_link': 'https://huggingface.co/test/model',
                'dataset_link': None,
                'code_link': None
            }]
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_parse_full_csv_line(self):
        """Test parsing CSV line with all three links."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            content = ("https://github.com/test/repo,"
                       "https://huggingface.co/datasets/test/dataset,"
                       "https://huggingface.co/test/model")
            f.write(content)
            temp_path = f.name

        try:
            result = parse_input(temp_path)
            expected = [{
                'model_link': 'https://huggingface.co/test/model',
                'dataset_link': 'https://huggingface.co/datasets/test/dataset',
                'code_link': 'https://github.com/test/repo'
            }]
            assert result == expected
        finally:
            os.unlink(temp_path)

    def test_parse_multiple_lines(self):
        """Test parsing multiple lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write("https://huggingface.co/test/model1\n")
            f.write(",,https://huggingface.co/test/model2\n")
            f.write("https://github.com/test/repo,,https://huggingface.co/test/model3")  # noqa: E501
            temp_path = f.name

        try:
            result = parse_input(temp_path)
            assert len(result) == 3
            assert result[0]['model_link'] == 'https://huggingface.co/test/model1'  # noqa: E501
            assert result[1]['model_link'] == 'https://huggingface.co/test/model2'  # noqa: E501
            assert result[2]['code_link'] == 'https://github.com/test/repo'
        finally:
            os.unlink(temp_path)

    def test_parse_whitespace_handling(self):
        """Test parsing with extra whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write(" https://github.com/test/repo , , "
                    "https://huggingface.co/test/model ")
            temp_path = f.name

        try:
            result = parse_input(temp_path)
            expected = [{
                'model_link': 'https://huggingface.co/test/model',
                'dataset_link': None,
                'code_link': 'https://github.com/test/repo'
            }]
            assert result == expected
        finally:
            os.unlink(temp_path)


class TestTimeEvaluation:
    """Test cases for evaluation timing functionality."""

    def test_time_evaluation_success(self):
        """Test timing a successful evaluation."""
        def dummy_eval(*args, **kwargs):
            return MetricResult(
                metric_type=MetricType.PERFORMANCE_CLAIMS,
                value=0.8,
                details={},
                latency_ms=100,
                error=None
            )

        result, exec_time = time_evaluation(dummy_eval)
        assert isinstance(result, MetricResult)
        assert result.value == 0.8
        assert exec_time >= 0

    def test_time_evaluation_with_args(self):
        """Test timing evaluation with arguments."""
        def dummy_eval(arg1, arg2, kwarg1=None):
            assert arg1 == "test1"
            assert arg2 == "test2"
            assert kwarg1 == "test3"
            return MetricResult(
                metric_type=MetricType.BUS_FACTOR,
                value=0.5,
                details={},
                latency_ms=50,
                error=None
            )

        result, exec_time = time_evaluation(dummy_eval, "test1", "test2",
                                            kwarg1="test3")
        assert result.value == 0.5
        assert exec_time >= 0

    def test_time_evaluation_exception(self):
        """Test timing evaluation that raises exception."""
        def failing_eval():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            time_evaluation(failing_eval)


class TestExtractModelName:
    """Test cases for model name extraction."""

    def test_extract_model_name_standard(self):
        """Test extracting model name from standard HF link."""
        link = "https://huggingface.co/microsoft/DialoGPT-medium"
        result = extract_model_name(link)
        assert result == "DialoGPT-medium"

    def test_extract_model_name_with_params(self):
        """Test extracting model name from link with parameters."""
        link = "https://huggingface.co/bert-base-uncased?tab=readme"
        result = extract_model_name(link)
        assert result == "bert-base-uncased"

    def test_extract_model_name_invalid_link(self):
        """Test extracting model name from invalid link."""
        link = "https://example.com/invalid/link"
        result = extract_model_name(link)
        assert result == "unknown_model"


class TestFormatSizeScore:
    """Test cases for size score formatting."""

    def test_format_size_score_full(self):
        """Test formatting size score at maximum value."""
        mock_result = Mock()
        mock_result.value = 1.0
        
        result = format_size_score(mock_result)
        
        assert "raspberry_pi" in result
        assert "jetson_nano" in result
        assert "desktop_pc" in result
        assert "aws_server" in result
        assert result["aws_server"] == 1.0

    def test_format_size_score_partial(self):
        """Test formatting size score at partial value."""
        mock_result = Mock()
        mock_result.value = 0.5
        
        result = format_size_score(mock_result)
        
        assert result["raspberry_pi"] == 0.1  # 0.5 * 0.2
        assert result["jetson_nano"] == 0.2   # 0.5 * 0.4
        assert result["desktop_pc"] == 0.4    # 0.5 * 0.8
        assert result["aws_server"] == 0.5

    def test_format_size_score_zero(self):
        """Test formatting size score at zero value."""
        mock_result = Mock()
        mock_result.value = 0.0
        
        result = format_size_score(mock_result)
        
        for platform in result.values():
            assert platform == 0.0


class TestFindMissingLinks:
    """Test cases for finding missing links functionality."""

    @patch('main.HuggingFaceAPIManager')
    def test_find_missing_links_no_missing(self, mock_hf_manager_class):
        """Test when no links are missing."""
        model_link = "https://huggingface.co/test/model"
        dataset_link = "https://huggingface.co/datasets/test/dataset"
        code_link = "https://github.com/test/repo"
        
        dataset_links, final_code_link = find_missing_links(
            model_link, dataset_link, code_link)
        
        assert dataset_link in dataset_links
        assert final_code_link == code_link

    @patch('main.HuggingFaceAPIManager')
    def test_find_missing_links_with_discovery(self, mock_hf_manager_class):
        """Test discovering missing links from model card."""
        # Setup mock
        mock_manager = Mock()
        mock_hf_manager_class.return_value = mock_manager
        mock_manager.model_link_to_id.return_value = "test/model"
        
        mock_model_info = Mock()
        mock_model_info.cardData = """
        Dataset: https://huggingface.co/datasets/discovered/dataset
        Code: https://github.com/discovered/repo
        """
        mock_manager.get_model_info.return_value = mock_model_info
        
        model_link = "https://huggingface.co/test/model"
        dataset_link = None
        code_link = None
        
        dataset_links, final_code_link = find_missing_links(
            model_link, dataset_link, code_link)
        
        assert any("discovered/dataset" in link for link in dataset_links)
        assert "discovered/repo" in final_code_link


class TestPrintTimingSummary:
    """Test cases for timing summary printing."""

    @patch('main.logging')
    def test_print_timing_summary(self, mock_logging):
        """Test printing timing summary."""
        results = {
            "Test Metric 1": (Mock(value=0.8), 1.5),
            "Test Metric 2": (Mock(value=0.6), 2.0)
        }
        total_time = 3.0
        
        print_timing_summary(results, total_time)
        
        # Verify logging was called
        assert mock_logging.info.called
        call_args = [call.args[0] for call in mock_logging.info.call_args_list]
        summary_calls = [arg for arg in call_args if "SUMMARY" in arg]
        assert len(summary_calls) > 0


@patch('main.ModelMetricService')
class TestRunEvaluations:
    """Test cases for running evaluations."""

    def test_run_evaluations_sequential(self, mock_service_class):
        """Test running evaluations sequentially."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Setup mock evaluations
        mock_result = MetricResult(
            metric_type=MetricType.PERFORMANCE_CLAIMS,
            value=0.7,
            details={},
            latency_ms=100,
            error=None
        )
        
        for method_name in ['EvaluatePerformanceClaims', 'EvaluateBusFactor',
                            'EvaluateSize', 'EvaluateRampUpTime']:
            setattr(mock_service, method_name, Mock(return_value=mock_result))
        
        mock_model_data = Mock()
        results = run_evaluations_sequential(mock_model_data)
        
        assert len(results) >= 4  # At least 4 evaluations
        for name, (result, exec_time) in results.items():
            assert isinstance(result, MetricResult)
            assert exec_time >= 0

    def test_run_evaluations_parallel(self, mock_service_class):
        """Test running evaluations in parallel."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Setup mock evaluations
        mock_result = MetricResult(
            metric_type=MetricType.PERFORMANCE_CLAIMS,
            value=0.7,
            details={},
            latency_ms=100,
            error=None
        )
        
        for method_name in ['EvaluatePerformanceClaims', 'EvaluateBusFactor',
                            'EvaluateSize', 'EvaluateRampUpTime']:
            setattr(mock_service, method_name, Mock(return_value=mock_result))
        
        mock_model_data = Mock()
        results = run_evaluations_parallel(mock_model_data, max_workers=2)
        
        assert len(results) >= 4  # At least 4 evaluations
        for name, (result, exec_time) in results.items():
            assert isinstance(result, MetricResult)
            assert exec_time >= 0