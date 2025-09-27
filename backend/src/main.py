from dotenv import load_dotenv
import logging
import os
from Controllers.Controller import Controller
from Services.Metric_Model_Service import ModelMetricService

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure logging to see debug info
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    fetcher = Controller()
    dataset_links = [
        "https://huggingface.co/datasets/xlangai/AgentNet",
        "https://huggingface.co/datasets/osunlp/UGround-V1-Data",
        "https://huggingface.co/datasets/xlangai/aguvis-stage2"
    ]
    code_link = "https://github.com/xlang-ai/OpenCUA"
    model_link = "https://huggingface.co/xlangai/OpenCUA-32B"

    model_data = fetcher.fetch(
        model_link,
        dataset_links=dataset_links,
        code_link=code_link
    )

    service = ModelMetricService()
    performance_claims = service.EvaluatePerformanceClaims(model_data)
    print(f"Performance Claims Score: {performance_claims.value}")

    bus_factor = service.EvaluateBusFactor(model_data)
    print(f"Bus Factor Score: {bus_factor.value}")

    size = service.EvaluateSize(model_data)
    print(f"Model Size Score: {size.value}")

    ramp_up_time = service.EvaluateRampUpTime(model_data)
    print(f"Ramp-Up Time Score: {ramp_up_time.value}")

    availability = service.EvaluateDatasetAndCodeAvailabilityScore(model_data)
    print(f"Availability Score: {availability.value}")

    codeQuality = service.EvaluateCodeQuality(model_data)
    print(f"Code Quality Score: {codeQuality.value}")

    datasetQuality = service.EvaluateDatasetsQuality(model_data)
    print(f"Dataset Quality Score: {datasetQuality.value}")
