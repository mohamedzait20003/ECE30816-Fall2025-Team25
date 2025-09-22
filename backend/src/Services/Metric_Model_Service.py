import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from lib.LLM_Manager import LLMManager
from lib.Metric_Result import MetricResult, MetricType
from Helpers import _parse_iso8601, _months_between
from Models.Manager_Models_Model import ModelManager


class ModelMetricService:
    def __init__(self) -> None:
        self.llm_manager = LLMManager()

    def EvaluateModel(
        self, model_description: str, dataset_description: str
    ) -> float:
        # TODO: Implement model evaluation logic
        pass

    def EvaluatePerformanceClaims(self, Model: ModelManager) -> MetricResult:
        def _compose_source_text(data: ModelManager) -> str:
            readme = ""
            path = getattr(data, "readme_path", None)
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        readme = fh.read()
                except Exception:
                    readme = ""
            card = ""
            card_obj = getattr(data, "card", None)
            if card_obj is not None:
                card = str(card_obj)
            text = (readme + "\n\n" + card).strip()
            if len(text) > 16000:
                text = text[:16000] + "\n\n...[truncated]..."
            return text

        def prepare_llm_prompt(data: ModelManager) -> str:
            assert isinstance(data, ModelManager)
            text = _compose_source_text(data)
            return (
                "You are evaluating a machine learning model card/README. "
                "Only use the provided text. Return STRICT JSON with these "
                "boolean fields and a short notes string:\n"
                "{\n"
                '  "has_benchmark_datasets": true|false,\n'
                '  "has_quantitative_results": true|false,\n'
                '  "has_baseline_or_sota_comparison": true|false,\n'
                '  "notes": "brief rationale"\n'
                "}\n\n"
                "Definitions:\n"
                "- Benchmark datasets: named datasets (e.g., SQuAD, GLUE, "
                "ImageNet, MMLU, etc.).\n"
                "- Quantitative results: numeric metrics or tables "
                "(accuracy, F1, BLEU, etc.).\n"
                "- Baseline/SoTA: comparison vs prior or state-of-the-art.\n\n"
                "=== BEGIN TEXT ===\n"
                f"{text}\n"
                "=== END TEXT ===\n"
            )

        def parse_llm_response(response: str) -> Dict[str, Any]:
            obj = json.loads(response)  # let it raise if bad JSON
            return {
                "has_benchmark_datasets": bool(
                    obj.get("has_benchmark_datasets", False)
                ),
                "has_quantitative_results": bool(
                    obj.get("has_quantitative_results", False)
                ),
                "has_baseline_or_sota_comparison": bool(
                    obj.get("has_baseline_or_sota_comparison", False)
                ),
                "notes": str(obj.get("notes", ""))[:400],
            }

        try:
            prompt = prepare_llm_prompt(Model)
            response = self.llm_manager.call_gemini_api(prompt)
            parsed = parse_llm_response(response.content)

            score = 0.0
            if parsed["has_benchmark_datasets"]:
                score += 0.3
            if parsed["has_quantitative_results"]:
                score += 0.4
            if parsed["has_baseline_or_sota_comparison"]:
                score += 0.3
            if score > 1.0:
                score = 1.0

            details = {"mode": "llm", **parsed}

            return MetricResult(
                metric_type=MetricType.PERFORMANCE_CLAIMS,
                value=score,
                details=details,
                latency_ms=0,
            )

        except Exception as exc:
            raise RuntimeError("LLM evaluation failed") from exc

    def EvaluateBusFactor(self, Model: ModelManager) -> MetricResult:
        def _contributors_score(contrib_count: int) -> float:
            if contrib_count >= 7:
                return 1.0
            if 4 <= contrib_count <= 6:
                return 0.7
            if 2 <= contrib_count <= 3:
                return 0.5
            if contrib_count == 1:
                return 0.3
            return 0.0

        def _recency_score(last_commit: Optional[datetime]) -> float:
            if last_commit is None:
                return 0.0
            now = datetime.now(timezone.utc)
            months = _months_between(now, last_commit)
            if months < 3.0:
                return 1.0
            score = 1.0 - 0.1 * (months - 3.0)
            if months > 12.0:
                return 0.0
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
            return score

        def _latest_commit_ts(data: ModelManager) -> Optional[datetime]:
            commits = getattr(data, "repo_commit_history", [])
            for item in commits:
                commit = item.get("commit", {})
                author = commit.get("author", {})
                ts = author.get("date")
                if isinstance(ts, str):
                    dt = _parse_iso8601(ts)
                    if dt is not None:
                        return dt
            return None

        def _contributors_count(data: ModelManager) -> int:
            contribs = getattr(data, "repo_contributors", [])
            if not isinstance(contribs, list):
                return 0
            return sum(
                1 for c in contribs
                if int(c.get("contributions", 0)) > 0
            )

        try:
            n_contrib = _contributors_count(Model)
            last_ts = _latest_commit_ts(Model)

            c_score = _contributors_score(n_contrib)
            r_score = _recency_score(last_ts)

            score = 0.7 * c_score + 0.3 * r_score
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0

            months = None
            if last_ts is not None:
                months = round(
                    _months_between(datetime.now(timezone.utc), last_ts), 2)

            details = {
                "contributors_count": n_contrib,
                "contributors_score": round(c_score, 3),
                "last_commit_months_ago": months,
                "recency_score": round(r_score, 3),
                "blend": "0.7*contributors + 0.3*recency",
            }

            return MetricResult(
                metric_type=MetricType.BUS_FACTOR,
                value=score,
                details=details,
                latency_ms=0,
            )

        except Exception as e:
            logging.error(f"Failed to evaluate bus factor: {e}")
            raise RuntimeError("Bus factor evaluation failed") from e

    def EvaluateSize(self, Model: ModelManager) -> MetricResult:
        def _size_metric(x: float) -> float:
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

        def _size_band_mb(
            x: float, a: float, b: float, c: float, d: float
        ) -> float:
            if x <= a:
                return 1.0
            if x <= b:
                return 0.6
            if x <= c:
                return 0.3
            if x <= d:
                return 0.1
            return 0.0

        try:
            if isinstance(Model.repo_metadata, dict):
                s = (
                    Model.repo_metadata.get("size_mb")
                    or Model.repo_metadata.get("size")
                )

                size_mb = 0.0
                if isinstance(s, str):
                    if s.lower().endswith("gb"):
                        try:
                            size_mb = float(s[:-2]) * 1024.0
                        except (ValueError, TypeError) as e:
                            logging.error(
                                f"Failed to parse GB size '{s}': {e}"
                            )
                            raise ValueError(
                                f"Invalid GB size format: {s}"
                            ) from e
                    else:
                        try:
                            size_mb = float(s)
                        except (ValueError, TypeError) as e:
                            logging.error(
                                f"Failed to parse MB size '{s}': {e}"
                            )
                            raise ValueError(
                                f"Invalid MB size format: {s}"
                            ) from e
                elif isinstance(s, (int, float)):
                    size_mb = float(s)

                r_pi = _size_metric(
                    _size_band_mb(size_mb, 200, 500, 1500, 2000)
                )

                j_nano = _size_metric(
                    _size_band_mb(size_mb, 400, 1500, 4000, 6000)
                )

                d_pc = _size_metric(
                    _size_band_mb(size_mb, 2000, 7000, 20000, 40000)
                )

                aws = _size_metric(
                    _size_band_mb(size_mb, 40000, 60000, 120000, 240000)
                )

                sizeScore = (r_pi + j_nano + d_pc + aws) / 4.0

                return MetricResult(
                    metric_type=MetricType.SIZE_SCORE,
                    value=sizeScore,
                    details={"derived_size_mb": size_mb},
                    latency_ms=0,
                )
            else:
                logging.warning("Model repo_metadata is not a dictionary")
                return MetricResult(
                    metric_type=MetricType.SIZE_SCORE,
                    value=0.0,
                    details={"error": "repo_metadata is not a dictionary"},
                    latency_ms=0,
                )

        except Exception as e:
            logging.error(f"Failed to evaluate model size: {e}")
            raise RuntimeError("Size evaluation failed") from e
