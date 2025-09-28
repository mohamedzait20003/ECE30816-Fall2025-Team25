import re
import os
import json
import pytest
import tempfile
from typing import List, Dict
from unittest.mock import patch, mock_open


def classify_url(url: str) -> str:
    """Classify URLs by their domain and path pattern."""
    if re.match(r"^https?://huggingface\.co/datasets/", url):
        return "dataset"
    elif re.match(r"^https?://huggingface\.co/", url):
        return "model"
    elif re.match(r"^https?://github\.com/", url):
        return "code"
    else:
        return "unknown"


def parse_url_file(filepath: str) -> List[Dict[str, str]]:
    """Parse a file containing URLs and classify them."""
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    category = classify_url(url)
                    results.append({"url": url, "type": category})
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return results


class TestClassifyUrl:
    """Test cases for classify_url function."""
    
    def test_classify_huggingface_dataset_url(self):
        """Test classification of HuggingFace dataset URLs."""
        url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
        result = classify_url(url)
        assert result == "dataset"
    
    def test_classify_huggingface_model_url(self):
        """Test classification of HuggingFace model URLs."""
        url = "https://huggingface.co/google-bert/bert-base-uncased"
        result = classify_url(url)
        assert result == "model"
    
    def test_classify_github_url(self):
        """Test classification of GitHub URLs."""
        url = "https://github.com/google-research/bert"
        result = classify_url(url)
        assert result == "code"
    
    def test_classify_unknown_url(self):
        """Test classification of unknown URLs."""
        url = "https://example.com/some-resource"
        result = classify_url(url)
        assert result == "unknown"
    
    def test_classify_http_url(self):
        """Test classification works with HTTP (not just HTTPS)."""
        url = "http://huggingface.co/datasets/test-dataset"
        result = classify_url(url)
        assert result == "dataset"
    
    @pytest.mark.parametrize("url,expected", [
        ("https://huggingface.co/datasets/test", "dataset"),
        ("https://huggingface.co/model-name", "model"),
        ("https://github.com/user/repo", "code"),
        ("https://other-site.com/resource", "unknown"),
        ("", "unknown"),
    ])
    def test_classify_url_parametrized(self, url, expected):
        """Parametrized test for multiple URL classifications."""
        result = classify_url(url)
        assert result == expected


class TestParseUrlFile:
    """Test cases for parse_url_file function."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_parse_valid_file(self, temp_file):
        """Test parsing a valid file with URLs."""
        test_urls = [
            "https://huggingface.co/datasets/bookcorpus/bookcorpus",
            "https://huggingface.co/google-bert/bert-base-uncased",
            "https://github.com/google-research/bert"
        ]
        
        # Write test URLs to temp file
        with open(temp_file, 'w', encoding='utf-8') as f:
            for url in test_urls:
                f.write(f"{url}\n")
        
        result = parse_url_file(temp_file)
        
        assert len(result) == 3
        assert result[0]['url'] == test_urls[0]
        assert result[0]['type'] == 'dataset'
        assert result[1]['url'] == test_urls[1]
        assert result[1]['type'] == 'model'
        assert result[2]['url'] == test_urls[2]
        assert result[2]['type'] == 'code'
    
    def test_parse_file_with_empty_lines(self, temp_file):
        """Test parsing file with empty lines."""
        content = (
            "https://huggingface.co/datasets/test\n"
            "\n"
            "\n"
            "https://github.com/test/repo\n"
        )
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = parse_url_file(temp_file)
        
        assert len(result) == 2
        assert result[0]['type'] == 'dataset'
        assert result[1]['type'] == 'code'
    
    def test_parse_nonexistent_file(self, capsys):
        """Test parsing a file that doesn't exist."""
        result = parse_url_file("nonexistent_file.txt")
        
        assert result == []
        captured = capsys.readouterr()
        assert "File not found: nonexistent_file.txt" in captured.out
    
    def test_parse_empty_file(self, temp_file):
        """Test parsing an empty file."""
        # File is already empty from fixture
        result = parse_url_file(temp_file)
        assert result == []
    
    @patch('builtins.open',
           mock_open(read_data="https://huggingface.co/test\n"))
    def test_parse_file_mock(self):
        """Test file parsing with mocked file operations."""
        result = parse_url_file("dummy_path.txt")
        
        assert len(result) == 1
        assert result[0]['url'] == "https://huggingface.co/test"
        assert result[0]['type'] == "model"
    
    def test_parse_csv_format(self, temp_file):
        """Test parsing CSV-like input format."""
        content = (
            "https://github.com/google-research/bert,"
            "https://huggingface.co/datasets/bookcorpus/bookcorpus,"
            "https://huggingface.co/google-bert/bert-base-uncased\n"
            ",,https://huggingface.co/parvk11/audience_classifier_model\n"
            ",,https://huggingface.co/openai/whisper-tiny/tree/main\n"
        )
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = parse_url_file(temp_file)
        
        # Should parse each line as a single URL (not CSV)
        assert len(result) == 3
        # First line is treated as one long URL (will be unknown)
        assert result[0]['type'] == 'unknown'
        # Other lines start with commas, so will be unknown
        assert result[1]['type'] == 'unknown'
        assert result[2]['type'] == 'unknown'


class TestMainIntegration:
    def test_integration_parse_and_classify(self):
        """Test integration between parse_url_file and classify_url."""
        # Create a test file with mixed URLs
        test_urls = [
            "https://github.com/user/repo",
            "not_a_url",
            "https://example.com",
            ""
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            for url in test_urls:
                f.write(f"{url}\n")
            temp_file = f.name
        
        try:
            # Parse URLs from file
            parsed_urls = parse_url_file(temp_file)
            
            # Classify each URL
            results = []
            for url in parsed_urls:
                classification = classify_url(url)
                results.append((url, classification))
            
            # Verify results
            assert len(results) == 4
            assert results[0][0] == "https://github.com/user/repo"
            assert results[0][1] == "github"  # GitHub URL
            assert results[1][0] == "not_a_url"
            assert results[1][1] == "other"  # Not a valid URL
            assert results[2][0] == "https://example.com"
            assert results[2][1] == "other"  # Valid URL but not GitHub
            assert results[3][0] == ""
            assert results[3][1] == "other"  # Empty string
            
        finally:
            os.unlink(temp_file)
    
    def test_json_output_format(self):
        """Test JSON output formatting."""
        sample_data = {
            'name': 'bert-base-uncased',
            'category': 'MODEL',
            'net_score': 0.95
        }
        
        json_output = json.dumps(sample_data,
                                 separators=(',', ':'),
                                 ensure_ascii=False)
        
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert parsed == sample_data
        
        # Verify compact format (no spaces after separators)
        assert ', ' not in json_output
        assert ': ' not in json_output


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_classify_url_edge_cases(self):
        """Test classify_url with edge cases."""
        # Empty string
        assert classify_url("") == "unknown"
        
        # None-like input (converted to string)
        assert classify_url(str(None)) == "unknown"
        
        # Malformed URLs
        assert classify_url("not-a-url") == "unknown"
        assert classify_url("ftp://example.com") == "unknown"
    
    def test_file_reading_errors(self, capsys):
        """Test various file reading error scenarios."""
        # Test with invalid file path characters (if supported by OS)
        result = parse_url_file("/invalid/path/file.txt")
        assert result == []
        
        # Check error message was printed
        captured = capsys.readouterr()
        assert "File not found" in captured.out
    
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_file_permission_error(self, mock_open):
        """Test handling of file permission errors."""
        # Should handle permission errors gracefully
        result = parse_url_file("protected_file.txt")
        assert result == []
    
    @patch('builtins.open', side_effect=IOError("I/O Error"))
    def test_file_io_error(self, mock_open):
        """Test handling of general I/O errors."""
        result = parse_url_file("problematic_file.txt")
        assert result == []


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])