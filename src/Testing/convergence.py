import time
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Configure logging for convergence testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvergenceStatus(Enum):
    """Status of convergence testing."""
    PENDING = "pending"
    CONVERGED = "converged"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ConvergenceResult:
    """Results of a convergence test."""
    status: ConvergenceStatus
    iterations: int
    final_value: Any = None
    error_message: Optional[str] = None
    execution_times: List[float] = field(default_factory=list)
    intermediate_values: List[Any] = field(default_factory=list)


class ConvergenceValidator:
    """Validates convergence of model fetching operations."""

    def __init__(self,
                 tolerance: float = 1e-6,
                 max_iterations: int = 100,
                 timeout_seconds: float = 60.0):
        """
        Initialize convergence validator.
        
        Args:
            tolerance: Tolerance for convergence detection
            max_iterations: Maximum number of iterations to test
            timeout_seconds: Timeout for convergence testing
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds

    def test_function_convergence(self,
                                  func: Callable,
                                  *args,
                                  **kwargs) -> ConvergenceResult:
        """
        Test if a function converges to a stable result.
        
        Args:
            func: Function to test for convergence
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            ConvergenceResult with test results
        """
        start_time = time.time()
        result = ConvergenceResult(
            status=ConvergenceStatus.PENDING,
            iterations=0
        )
        
        previous_value = None
        
        try:
            for i in range(self.max_iterations):
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    result.status = ConvergenceStatus.TIMEOUT
                    result.error_message = "Convergence test timed out"
                    break
                
                # Execute function and measure time
                func_start_time = time.time()
                current_value = func(*args, **kwargs)
                execution_time = time.time() - func_start_time
                
                result.execution_times.append(execution_time)
                result.intermediate_values.append(current_value)
                result.iterations = i + 1
                
                # Check convergence
                if previous_value is not None:
                    if self._check_convergence(previous_value, current_value):
                        result.status = ConvergenceStatus.CONVERGED
                        result.final_value = current_value
                        logger.info(
                            f"Function converged after {i + 1} iterations"
                        )
                        break
                
                previous_value = current_value
            
            else:
                # Maximum iterations reached without convergence
                result.status = ConvergenceStatus.FAILED
                result.error_message = (
                    "Maximum iterations reached without convergence"
                )
                result.final_value = (
                    current_value if 'current_value' in locals() else None
                )
                
        except Exception as e:
            result.status = ConvergenceStatus.FAILED
            result.error_message = (
                f"Exception during convergence test: {str(e)}"
            )
            logger.error(f"Convergence test failed with exception: {e}")
        
        return result

    def _check_convergence(self, value1: Any, value2: Any) -> bool:
        """
        Check if two values are considered converged.
        
        Args:
            value1: First value to compare
            value2: Second value to compare
            
        Returns:
            True if values are converged, False otherwise
        """
        try:
            # Handle numeric values
            if (isinstance(value1, (int, float)) and
                    isinstance(value2, (int, float))):
                return abs(value1 - value2) <= self.tolerance
            
            # Handle string values
            if isinstance(value1, str) and isinstance(value2, str):
                return value1 == value2
            
            # Handle dictionary values (for model metadata)
            if isinstance(value1, dict) and isinstance(value2, dict):
                return self._compare_dicts(value1, value2)
            
            # Handle list values
            if isinstance(value1, list) and isinstance(value2, list):
                return self._compare_lists(value1, value2)
            
            # Handle object comparison by comparing string representations
            return str(value1) == str(value2)
            
        except Exception as e:
            logger.warning(f"Error comparing values: {e}")
            return False

    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> bool:
        """Compare two dictionaries for convergence."""
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        
        for key in dict1:
            if not self._check_convergence(dict1[key], dict2[key]):
                return False
        
        return True

    def _compare_lists(self, list1: List, list2: List) -> bool:
        """Compare two lists for convergence."""
        if len(list1) != len(list2):
            return False
        
        for item1, item2 in zip(list1, list2):
            if not self._check_convergence(item1, item2):
                return False
        
        return True


class ModelFetcherConvergenceTest:
    """Specific convergence tests for ModelFetcher operations."""

    def __init__(self, validator: ConvergenceValidator):
        """
        Initialize ModelFetcher convergence test.
        
        Args:
            validator: ConvergenceValidator instance to use
        """
        self.validator = validator

    def test_model_info_consistency(self,
                                    fetcher,
                                    model_link: str) -> ConvergenceResult:
        """
        Test if model info fetching is consistent across multiple calls.
        
        Args:
            fetcher: ModelFetcher instance
            model_link: Model link to test
            
        Returns:
            ConvergenceResult for consistency test
        """
        def fetch_model_info():
            """Helper function to fetch model info."""
            model_data = fetcher.fetch_model(model_link)
            return {
                'id': model_data.id,
                'info_id': model_data.info.id if model_data.info else None,
                'readme_exists': model_data.readme_path is not None
            }
        
        return self.validator.test_function_convergence(fetch_model_info)

    def test_dataset_fetching_stability(self,
                                        fetcher,
                                        model_link: str,
                                        dataset_links: List[str]
                                        ) -> ConvergenceResult:
        """
        Test stability of dataset information fetching.
        
        Args:
            fetcher: ModelFetcher instance
            model_link: Model link to test
            dataset_links: List of dataset links
            
        Returns:
            ConvergenceResult for dataset stability test
        """
        def fetch_dataset_info():
            """Helper function to fetch dataset info."""
            model_data = fetcher.fetch_model(
                model_link, dataset_links=dataset_links
            )
            return {
                'dataset_count': len(model_data.dataset_ids),
                'dataset_ids': sorted(model_data.dataset_ids),
                'info_count': len(model_data.dataset_infos),
                'card_count': len(model_data.dataset_cards)
            }
        
        return self.validator.test_function_convergence(fetch_dataset_info)

    def test_github_repo_consistency(self,
                                     fetcher,
                                     model_link: str,
                                     code_link: str) -> ConvergenceResult:
        """
        Test consistency of GitHub repository information fetching.
        
        Args:
            fetcher: ModelFetcher instance
            model_link: Model link to test
            code_link: GitHub repository link
            
        Returns:
            ConvergenceResult for GitHub consistency test
        """
        def fetch_repo_info():
            """Helper function to fetch repository info."""
            model_data = fetcher.fetch_model(model_link, code_link=code_link)
            return {
                'has_metadata': bool(model_data.repo_metadata),
                'contents_count': len(model_data.repo_contents),
                'contributors_count': len(model_data.repo_contributors),
                'commits_count': len(model_data.repo_commit_history)
            }
        
        return self.validator.test_function_convergence(fetch_repo_info)


def run_convergence_test_suite(
        fetcher, test_config: Dict[str, Any]
) -> Dict[str, ConvergenceResult]:
    """
    Run a complete convergence test suite for ModelFetcher.
    
    Args:
        fetcher: ModelFetcher instance to test
        test_config: Configuration dictionary with test parameters
        
    Returns:
        Dictionary of test results
    """
    validator = ConvergenceValidator(
        tolerance=test_config.get('tolerance', 1e-6),
        max_iterations=test_config.get('max_iterations', 10),
        timeout_seconds=test_config.get('timeout_seconds', 30.0)
    )
    
    convergence_tester = ModelFetcherConvergenceTest(validator)
    results = {}
    
    # Test model info consistency
    if 'model_link' in test_config:
        logger.info("Testing model info consistency...")
        results['model_info_consistency'] = (
            convergence_tester.test_model_info_consistency(
                fetcher, test_config['model_link']
            )
        )
    
    # Test dataset fetching stability
    if 'model_link' in test_config and 'dataset_links' in test_config:
        logger.info("Testing dataset fetching stability...")
        results['dataset_stability'] = (
            convergence_tester.test_dataset_fetching_stability(
                fetcher, test_config['model_link'],
                test_config['dataset_links']
            )
        )
    
    # Test GitHub repo consistency
    if 'model_link' in test_config and 'code_link' in test_config:
        logger.info("Testing GitHub repo consistency...")
        results['github_consistency'] = (
            convergence_tester.test_github_repo_consistency(
                fetcher, test_config['model_link'], test_config['code_link']
            )
        )
    
    return results


def print_convergence_results(results: Dict[str, ConvergenceResult]) -> None:
    """
    Print convergence test results in a readable format.
    
    Args:
        results: Dictionary of convergence results to print
    """
    print("\n" + "="*60)
    print("CONVERGENCE TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        print(f"  Status: {result.status.value}")
        print(f"  Iterations: {result.iterations}")
        
        if result.execution_times:
            avg_time = (
                sum(result.execution_times) / len(result.execution_times)
            )
            print(f"  Average execution time: {avg_time:.4f}s")
        
        if result.error_message:
            print(f"  Error: {result.error_message}")
        
        if result.status == ConvergenceStatus.CONVERGED:
            print("  ✓ Test PASSED - Function converged")
        else:
            print("  ✗ Test FAILED - Function did not converge")


# Example usage and test configuration
SAMPLE_TEST_CONFIG = {
    'tolerance': 1e-6,
    'max_iterations': 5,
    'timeout_seconds': 30.0,
    'model_link': 'https://huggingface.co/xlangai/OpenCUA-32B',
    'dataset_links': [
        'https://huggingface.co/datasets/xlangai/AgentNet'
    ],
    'code_link': 'https://github.com/xlang-ai/OpenCUA'
}


if __name__ == "__main__":
    # This would be used for standalone testing
    print("Convergence testing utilities loaded.")
    print("Use run_convergence_test_suite() to execute tests.")