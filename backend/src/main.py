from dotenv import load_dotenv
import logging
import time
import os
import csv
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
        ("Availability", service.EvaluateDatasetAndCodeAvailabilityScore),
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


def parse_input(file_path):
    """
    Parse input file with lines like:
    [code_link], [dataset_link], model_link
    Returns a list of dicts with keys: dataset_link, code_link, model_link
    """
    jobs = []
    # If the file path is relative, resolve it relative to the project root
    if not os.path.isabs(file_path):
        # Get the project root directory (two levels up from this file)
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        file_path = os.path.join(project_root, file_path)
        file_path = os.path.normpath(file_path)
    
    with open(file_path, encoding='utf-8') as f:
        for row in csv.reader(f):
            # Remove whitespace and ignore empty fields
            row = [x.strip() for x in row if x.strip()]
            if not row:
                continue
            # Always take the last field as model_link
            model_link = row[-1]
            code_link = row[0] if len(row) > 1 else None
            dataset_link = row[1] if len(row) > 2 else None
            jobs.append({
                'model_link': model_link,
                'dataset_link': dataset_link,
                'code_link': code_link
            })
    return jobs


def find_missing_links(model_link, dataset_link, code_link):
    """
    If dataset_link or code_link is missing, try to find them from the model card.
    Uses HuggingFace API to get model info and parse for links.
    """
    from lib.HuggingFace_API_Manager import HuggingFaceAPIManager
    import re
    
    discovered_datasets = []
    discovered_code = None
    
    try:
        # Initialize HuggingFace API manager
        hf_manager = HuggingFaceAPIManager()
        
        # Extract model ID from the link
        model_id = hf_manager.model_link_to_id(model_link)
        print(f"  Searching for links in model: {model_id}")
        
        # Get model information
        model_info = hf_manager.get_model_info(model_id)
        
        # Check model card text for links
        if hasattr(model_info, 'cardData') and model_info.cardData:
            card_text = str(model_info.cardData)
            
            # Look for dataset links in model card
            dataset_patterns = [
                r'https://huggingface\.co/datasets/([^/\s]+/[^/\s]+)',
                r'huggingface\.co/datasets/([^/\s]+/[^/\s]+)',
                r'datasets/([^/\s\)]+/[^/\s\)]+)',
            ]
            
            for pattern in dataset_patterns:
                matches = re.findall(pattern, card_text, re.IGNORECASE)
                for match in matches:
                    if not match.startswith('http'):
                        dataset_url = (f"https://huggingface.co/datasets/"
                                       f"{match}")
                    else:
                        dataset_url = match
                    if dataset_url not in discovered_datasets:
                        discovered_datasets.append(dataset_url)
            
            # Look for GitHub/code repository links
            code_patterns = [
                r'https://github\.com/([^/\s\)]+/[^/\s\)]+)',
                r'github\.com/([^/\s\)]+/[^/\s\)]+)',
                r'\[.*?\]\(https://github\.com/([^/\s\)]+/[^/\s\)]+)\)',
                r'repo:\s*([^/\s]+/[^/\s]+)',
                r'code:\s*https://github\.com/([^/\s\)]+/[^/\s\)]+)',
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, card_text, re.IGNORECASE)
                if matches and not discovered_code:
                    match = matches[0]  # Take the first one
                    # Clean up the match (remove trailing punctuation)
                    match = match.rstrip('.,;)')
                    if not match.startswith('http'):
                        discovered_code = f"https://github.com/{match}"
                    else:
                        discovered_code = match
                    break
        
        # Also check model tags and metadata
        if hasattr(model_info, 'tags') and model_info.tags:
            for tag in model_info.tags:
                if 'dataset:' in tag:
                    dataset_name = tag.replace('dataset:', '').strip()
                    if '/' in dataset_name:
                        dataset_url = (f"https://huggingface.co/datasets/"
                                       f"{dataset_name}")
                        if dataset_url not in discovered_datasets:
                            discovered_datasets.append(dataset_url)
        
        # Check model info for repository URL
        if hasattr(model_info, 'modelId') and not discovered_code:
            # Try to find associated GitHub repo through common naming
            model_id_parts = model_info.modelId.split('/')
            if len(model_id_parts) == 2:
                org, model_name = model_id_parts
                # Try common GitHub URL patterns
                # Remove size/version suffixes (e.g., -medium, -large, -32B)
                base_name = re.sub(r'-(?:small|medium|large|xl|xxl|\d+[BMG]?)$',
                                   '', model_name, flags=re.IGNORECASE)
                
                potential_repos = [
                    f"https://github.com/{org}/{base_name}",
                    f"https://github.com/{org}/{model_name}",
                    f"https://github.com/{org}/{model_name.lower()}",
                    f"https://github.com/{org.lower()}/{base_name}",
                ]
                # We'll check if these exist later, for now just take first
                discovered_code = potential_repos[0]
        
    except Exception as e:
        print(f"  Warning: Could not fetch model info for {model_link}: {e}")
    
    # Use provided links first, fall back to discovered links
    final_dataset_links = []
    if dataset_link and dataset_link.strip():
        final_dataset_links.append(dataset_link.strip())
    final_dataset_links.extend(discovered_datasets)
    
    final_code_link = code_link
    if not final_code_link or not final_code_link.strip():
        final_code_link = discovered_code
    
    # Report what we found
    if discovered_datasets:
        dataset_preview = ', '.join(discovered_datasets[:3])
        more_text = '...' if len(discovered_datasets) > 3 else ''
        print(f"  Found {len(discovered_datasets)} dataset link(s): "
              f"{dataset_preview}{more_text}")
    if discovered_code:
        print(f"  Found code repository: {discovered_code}")
    
    return final_dataset_links, final_code_link


def run_batch_evaluation(input_file):
    jobs = parse_input(input_file)
    for i, job in enumerate(jobs, 1):
        print(f"\n=== Evaluating Model {i}/{len(jobs)} ===")
        model_link = job['model_link']
        print(f"Model: {model_link}")
        
        dataset_links, code_link = find_missing_links(
            model_link, job.get('dataset_link'), job.get('code_link'))
        
        # Debug: Show what links we're using
        print(f"  Dataset links: {dataset_links}")
        print(f"  Code link: {code_link}")
        
        fetcher = Controller()
        print("Fetching model data...")
        model_data = fetcher.fetch(
            model_link,
            dataset_links=dataset_links if dataset_links else [],
            code_link=code_link
        )
        print("Model data fetched successfully!")
        
        # Debug: Show what data we actually got
        repo_keys = (list(getattr(model_data, 'repo_metadata', {}).keys())
                     if hasattr(model_data, 'repo_metadata') else 'None')
        print(f"  Repo metadata keys: {repo_keys}")
        print(f"  Contributors count: "
              f"{len(getattr(model_data, 'repo_contributors', []))}")
        print(f"  Commits count: "
              f"{len(getattr(model_data, 'repo_commit_history', []))}")
        print(f"  Dataset IDs: {getattr(model_data, 'dataset_ids', [])}")
        
        start_time = time.perf_counter()
        results = run_evaluations_parallel(model_data, max_workers=4)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print_timing_summary(results, total_time)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith('.txt'):
        run_batch_evaluation(sys.argv[1])
    else:
        # error: please provide a .txt file with model links
        print("Usage: python main.py sample_input.txt")
