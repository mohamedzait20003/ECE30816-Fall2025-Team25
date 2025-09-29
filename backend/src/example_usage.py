"""
Example usage script demonstrating timing and parallel evaluation execution.
"""
import time
from Controllers.Controller import Controller
from main import run_evaluations_sequential, run_evaluations_parallel


def demo_timing():
    """Demonstrate timing functionality with both sequential and parallel."""
    # Sample data setup
    fetcher = Controller()
    dataset_links = [
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://huggingface.co/datasets/osunlp/UGround-V1-Data",
        "https://huggingface.co/datasets/xlangai/aguvis-stage2"
    ]
    code_link = "https://github.com/xlang-ai/OpenCUA"
    model_link = "https://huggingface.co/xlangai/OpenCUA-32B"

    print("Fetching model data...")
    model_data = fetcher.fetch(
        model_link,
        dataset_links=dataset_links,
        code_link=code_link
    )
    print("Model data fetched!")

    # Sequential timing example
    print("\n=== SEQUENTIAL EXECUTION ===")
    start_time = time.perf_counter()
    run_evaluations_sequential(model_data)
    seq_time = time.perf_counter() - start_time
    print(f"Total sequential time: {seq_time:.3f}s")

    # Parallel timing example
    print("\n=== PARALLEL EXECUTION (4 workers) ===")
    start_time = time.perf_counter()
    run_evaluations_parallel(model_data, max_workers=4)
    par_time = time.perf_counter() - start_time
    print(f"Total parallel time: {par_time:.3f}s")

    # Show speedup
    if par_time > 0:
        speedup = seq_time / par_time
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    demo_timing()
