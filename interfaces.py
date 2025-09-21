from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any, Dict, Optional, TypedDict, Union, cast


# ---------- Helpers ----------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class SizeScore(TypedDict):
    raspberry_pi: float
    jetson_nano: float
    desktop_pc: float
    aws_server: float


# ---------- Metric typing ----------

class MetricType(Enum):
    """Enumeration of supported metric types (names mirror NDJSON keys)."""
    SIZE_SCORE = "size_score"
    LICENSE = "license"
    RAMP_UP_TIME = "ramp_up_time"
    BUS_FACTOR = "bus_factor"
    DATASET_AND_CODE_SCORE = "dataset_and_code_score"
    DATASET_QUALITY = "dataset_quality"
    CODE_QUALITY = "code_quality"
    PERFORMANCE_CLAIMS = "performance_claims"


ValueType = Union[float, SizeScore]


@dataclass(frozen=True)
class MetricResult:
    """
    Result of a metric calculation.

    Attributes:
        metric_type: Type of metric calculated
        value: float (0..1) OR SizeScore mapping, depending on metric
        details: Additional details about the calculation
        latency_ms: Time to calculate metric, integer milliseconds (rounded)
        error: Error message if calculation failed (None if success)
    """
    metric_type: MetricType
    value: ValueType
    details: Dict[str, Any]
    latency_ms: int
    error: Optional[str] = None

    def is_success(self) -> bool:
        return self.error is None

    def __post_init__(self) -> None:
        # Validate ranges
        if isinstance(self.value, float):
            if not (0.0 <= self.value <= 1.0):
                raise ValueError(f"value must be in [0,1], got {self.value}")
        else:
            # SizeScore dict
            ss = cast(SizeScore, self.value)
            required_keys = {"raspberry_pi", "jetson_nano",
                             "desktop_pc", "aws_server"}
            missing = required_keys - set(ss.keys())
            if missing:
                raise ValueError(f"SizeScore missing keys: {missing}")
            for k in ("raspberry_pi", "jetson_nano",
                      "desktop_pc", "aws_server"):
                v = ss[k]  # type: float
                if not (0.0 <= v <= 1.0):
                    raise
                ValueError(f"SizeScore[{k}] must be in [0,1], got {v}")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")


# ---------- Input data containers ----------

@dataclass(frozen=True)
class ModelData:
    """
    Container for model metadata from various sources.
    - model_metadata: dict from HF Hub model repo or parsed files
    - dataset_metadata: dict for linked dataset(s)
    - code_metadata: dict for linked code repo(s)
    - url: original URL analyzed
    """
    model_metadata: Dict[str, Any]
    dataset_metadata: Optional[Dict[str, Any]] = None
    code_metadata: Optional[Dict[str, Any]] = None
    url: str = ""


@dataclass(frozen=True)
class DatasetData:
    """
    Container for dataset metadata from HF Hub.
    - dataset_metadata: dataset card/metadata dict
    - url: original URL analyzed
    """
    dataset_metadata: Dict[str, Any]
    url: str = ""


InputData = Union[ModelData, DatasetData]


# ---------- Task interfaces ----------

class MetricCalculationTask(ABC):
    """
    Base class for all metric calculation tasks.

    Implementors should override:
      - metric_type (property)
      - weight (property)  # used by NetScore
      - calculate(data) -> MetricResult  # pure compute, without timing/error
        handling

    Consumers should call .run(data) to get timed, robust results.
    """

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        ...

    @property
    @abstractmethod
    def weight(self) -> float:
        """Weight in NetScore (0..1). Sum across metrics should be 1.0 in
           scorer."""
        ...

    @abstractmethod
    def calculate(self, data: InputData) -> MetricResult:
        """
        Compute the metric. Implementations should:
          - clamp outputs to [0,1] (or SizeScore values to [0,1])
          - fill details with useful traces (e.g., counts, thresholds used)
          - set latency_ms to a placeholder (will be overwritten by run()) OR 0
          - leave error=None on success
        """
        ...

    def validate_input(self, data: InputData) -> bool:
        """Override per metric if you require certain fields."""
        return data is not None

    def get_description(self) -> str:
        return f"Calculates {self.metric_type.value}"

    # Unified timing & error boundary
    def run(self, data: InputData) -> MetricResult:
        if not self.validate_input(data):
            return MetricResult(
                metric_type=self.metric_type,
                value=0.0,
                details={"reason": "invalid_input"},
                latency_ms=0,
                error="Invalid input for this metric.",
            )
        t0 = perf_counter()
        try:
            result = self.calculate(data)
            # Overwrite latency with measured ms
            elapsed_ms = int(round((perf_counter() - t0) * 1000))
            return MetricResult(
                metric_type=result.metric_type,
                value=result.value,
                details=result.details,
                latency_ms=elapsed_ms,
                error=result.error,
            )
        except Exception as exc:
            elapsed_ms = int(round((perf_counter() - t0) * 1000))
            return MetricResult(
                metric_type=self.metric_type,
                value=0.0,
                details={"exception": type(exc).__name__},
                latency_ms=elapsed_ms,
                error=str(exc),
            )


class MetricCalculationTaskLLM(MetricCalculationTask):
    """
    Base class for metrics that rely on LLM assistance (e.g., README scoring).
    """

    def __init__(self, ai_agent_service: Optional[Any] = None) -> None:
        self.ai_agent_service = ai_agent_service

    @abstractmethod
    def prepare_llm_prompt(self, data: InputData) -> str:
        ...

    @abstractmethod
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        ...

    def validate_input(self, data: InputData) -> bool:
        if not super().validate_input(data):
            return False
        if isinstance(data, ModelData):
            return bool(
                data.model_metadata or data.dataset_metadata
                or data.code_metadata
            )
        if isinstance(data, DatasetData):
            return bool(data.dataset_metadata)
        return False


# ---------- Factory ----------

class MetricTaskFactory:
    """Factory for instantiating metric tasks with optional dependencies."""

    def __init__(self, ai_agent_service: Optional[Any] = None) -> None:
        self.ai_agent_service = ai_agent_service

    def create_metric_task(self,
                           metric_type: MetricType) -> MetricCalculationTask:
        # Wire your real implementations here. Example stub for size:
        if metric_type == MetricType.SIZE_SCORE:
            return _ExampleSizeTask()
        # elif metric_type == MetricType.LICENSE:
        #     return LicenseTask()
        # elif metric_type == MetricType.RAMP_UP_TIME:
        #     return RampUpTimeTask(self.ai_agent_service)
        # ...
        raise NotImplementedError(f"Metric {metric_type} not implemented.")

    def create_all_metrics(self) -> Dict[MetricType, MetricCalculationTask]:
        out: Dict[MetricType, MetricCalculationTask] = {}
        for m in MetricType:
            try:
                out[m] = self.create_metric_task(m)
            except NotImplementedError:
                continue
        return out


# ---------- Tiny example implementation ----------

class _ExampleSizeTask(MetricCalculationTask):
    """Toy size metric just to demonstrate the interface."""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.SIZE_SCORE

    @property
    def weight(self) -> float:
        return 0.08  # example; your scorer will enforce the total

    def calculate(self, data: InputData) -> MetricResult:
        size_mb = 0.0
        if isinstance(data, ModelData):
            s = (
                data.model_metadata.get("size_mb")
                or data.model_metadata.get("size")
                )
            if isinstance(s, str):
                if s.lower().endswith("gb"):
                    try:
                        size_mb = float(s[:-2]) * 1024.0
                    except Exception:
                        size_mb = 0.0
                else:
                    try:
                        size_mb = float(s)
                    except Exception:
                        size_mb = 0.0
            elif isinstance(s, (int, float)):
                size_mb = float(s)

        def band_mb(x: float, a: float, b: float, c: float, d: float) -> float:
            if x <= a:
                return 1.0
            if x <= b:
                return 0.6
            if x <= c:
                return 0.3
            if x <= d:
                return 0.1
            return 0.0

        r_pi = clamp01(band_mb(size_mb, 200, 500, 1500, 2000))
        j_nano = clamp01(band_mb(size_mb, 400, 1500, 4000, 6000))
        d_pc = clamp01(band_mb(size_mb, 2000, 7000, 20000, 40000))
        aws = clamp01(band_mb(size_mb, 40000, 60000, 120000, 240000))

        size_score: SizeScore = {
            "raspberry_pi": r_pi,
            "jetson_nano": j_nano,
            "desktop_pc": d_pc,
            "aws_server": aws,
        }

        return MetricResult(
            metric_type=self.metric_type,
            value=size_score,
            details={"derived_size_mb": size_mb},
            latency_ms=0,  # overwritten by .run()
        )

# ---------- Example run ----------


if __name__ == "__main__":
    sample_model_data = ModelData(
        model_metadata={"name": "test-model", "size": "1GB"},
        dataset_metadata={"name": "test-dataset"},
        code_metadata={"repo": "github.com/test/repo"},
        url="https://huggingface.co/test-model",
    )

    task = _ExampleSizeTask()
    result = task.run(sample_model_data)
    print(result)
    print("success?", result.is_success())
