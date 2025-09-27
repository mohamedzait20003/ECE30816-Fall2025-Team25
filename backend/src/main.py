from dotenv import load_dotenv
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Tuple, Dict
from Controllers.Controller import Controller
from Services.Metric_Model_Service import ModelMetricService
from lib.Metric_Result import MetricResult

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure logging to see debug info
logging.basicConfig(level=logging.INFO)


def time_evaluation(eval_func: Callable, *args, **kwargs) -> \
        Tuple[MetricResult, float]:
    """
    Times a single evaluation function and returns the result and execution
    time.
    
    Args:
        eval_func: The evaluation function to time
        *args: Arguments to pass to the evaluation function
        **kwargs: Keyword arguments to pass to the evaluation function
    
    Returns:
        Tuple containing (MetricResult, execution_time_in_seconds)
    """
    start_time = time.perf_counter()
    try:
        result = eval_func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    except Exception as e:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logging.error(f"Error in evaluation {eval_func.__name__}: {e}")
        raise


def run_evaluations_sequential(model_data) -> \
        Dict[str, Tuple[MetricResult, float]]:
    """
    Run all evaluations sequentially and time each one.
    
    Args:
        model_data: The model data to evaluate
        
    Returns:
        Dictionary mapping evaluation names to (result, time) tuples
    """
    service = ModelMetricService()
    results = {}
    
    # Define all evaluations
    evaluations = [
        ("Performance Claims", service.EvaluatePerformanceClaims),
        ("Bus Factor", service.EvaluateBusFactor),
        ("Size", service.EvaluateSize),
        ("Ramp-Up Time", service.EvaluateRampUpTime),
        ("Availability", service.EvaluateAvailability),
        ("Code Quality", service.EvaluateCodeQuality),
        ("Dataset Quality", service.EvaluateDatasetsQuality),
        ("License", service.EvaluateLicense)
    ]
    
    print("Running evaluations sequentially...")
    print("-" * 50)
    
    for name, eval_func in evaluations:
        print(f"Starting: {name}")
        result, exec_time = time_evaluation(eval_func, model_data)
        results[name] = (result, exec_time)
        print(f"Completed: {name} - Score: {result.value:.3f} - "
              f"Time: {exec_time:.3f}s")
    
    return results


def run_evaluations_parallel(model_data, max_workers: int = 4) -> \
        Dict[str, Tuple[MetricResult, float]]:
    """
    Run all evaluations in parallel using ThreadPoolExecutor and time each one.
    
    Args:
        model_data: The model data to evaluate
        max_workers: Maximum number of worker threads (default: 4)
        
    Returns:
        Dictionary mapping evaluation names to (result, time) tuples
    """
    service = ModelMetricService()
    results = {}
    
    # Define all evaluations
    evaluations = [
        ("Performance Claims", service.EvaluatePerformanceClaims),
        ("Bus Factor", service.EvaluateBusFactor),
        ("Size", service.EvaluateSize),
        ("Ramp-Up Time", service.EvaluateRampUpTime),
        ("Availability", service.EvaluateDatasetAndCodeAvailabilityScore),
        ("Code Quality", service.EvaluateCodeQuality),
        ("Dataset Quality", service.EvaluateDatasetsQuality),
        ("License", service.EvaluateLicense)
    ]
    
    print(f"Running evaluations in parallel (max_workers={max_workers})...")
    print("-" * 50)
    
    # Submit all tasks to the thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluations
        future_to_name = {}
        for name, eval_func in evaluations:
            future = executor.submit(time_evaluation, eval_func, model_data)
            future_to_name[future] = name
            print(f"Submitted: {name}")
        
        # Collect results as they complete
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result, exec_time = future.result()
                results[name] = (result, exec_time)
                print(f"Completed: {name} - Score: {result.value:.3f} - "
                      f"Time: {exec_time:.3f}s")
            except Exception as e:
                print(f"Failed: {name} - Error: {e}")
                # You might want to store the error in results or handle
                # it differently
                
    return results


def print_timing_summary(results: Dict[str, Tuple[MetricResult, float]],
                         total_time: float):
    """
    Print a summary of all evaluation results and timing information.
    
    Args:
        results: Dictionary mapping evaluation names to (result, time) tuples
        total_time: Total execution time for all evaluations
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    total_eval_time = 0.0
    for name, (result, exec_time) in results.items():
        print(f"{name:<20}: Score = {result.value:.3f}, "
              f"Time = {exec_time:.3f}s")
        total_eval_time += exec_time
    
    print("-" * 60)
    print(f"{'Total Eval Time':<20}: {total_eval_time:.3f}s")
    print(f"{'Wall Clock Time':<20}: {total_time:.3f}s")
    
    if total_time > 0:
        efficiency = (total_eval_time / total_time) * 100
        print(f"{'Parallelism Efficiency':<20}: {efficiency:.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    fetcher = Controller()
    dataset_links = [
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://huggingface.co/datasets/osunlp/UGround-V1-Data",
        "https://huggingface.co/datasets/xlangai/aguvis-stage2"
    ]
    code_link = "https://github.com/xlang-ai/OpenCUA"
    model_link = "https://huggingface.co/xlangai/OpenCUA-32B"

    # Fetch model data
    print("Fetching model data...")
    model_data = fetcher.fetch(
        model_link,
        dataset_links=dataset_links,
        code_link=code_link
    )
    print("Model data fetched successfully!")

    # Choose execution mode
    run_parallel = True  # Set to False for sequential execution
    
    if run_parallel:
        # Run evaluations in parallel
        start_time = time.perf_counter()
        results = run_evaluations_parallel(model_data, max_workers=4)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print_timing_summary(results, total_time)
    else:
        # Run evaluations sequentially
        start_time = time.perf_counter()
        results = run_evaluations_sequential(model_data)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print_timing_summary(results, total_time)
