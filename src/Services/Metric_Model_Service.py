import json
import logging
import concurrent.futures
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from Models import Model
from lib.LLM_Manager import LLMManager
from Helpers import _parse_iso8601, _months_between, MetricResult, MetricType


class ModelMetricService:
    def __init__(self) -> None:
        self.llm_manager = LLMManager()
    
    def _extract_size_scores(self,
                             size_result: MetricResult) -> Dict[str, float]:
        """Extract device-specific size scores from MetricResult details."""
        if (hasattr(size_result, 'details') and
                isinstance(size_result.details, dict)):
            details = size_result.details
            return {
                "raspberry_pi": round(details.get("raspberry_pi", 0.0), 2),
                "jetson_nano": round(details.get("jetson_nano", 0.0), 2),
                "desktop_pc": round(details.get("desktop_pc", 0.0), 2),
                "aws_server": round(details.get("aws_server", 0.0), 2)
            }
        else:
            # Fallback if details are not available
            return {
                "raspberry_pi": 0.0,
                "jetson_nano": 0.0,
                "desktop_pc": 0.0,
                "aws_server": 0.0
            }

    def EvaluateModel(self, Data: Model) -> Dict[str, Any]:
        results = {}
        evaluation_tasks = {
            'ramp_up_time': self.EvaluateRampUpTime,
            'bus_factor': self.EvaluateBusFactor,
            'performance_claims': self.EvaluatePerformanceClaims,
            'license': self.EvaluateLicense,
            'size_score': self.EvaluateSize,
            'dataset_and_code_score': (
                self.EvaluateDatasetAndCodeAvailabilityScore
            ),
            'dataset_quality': self.EvaluateDatasetsQuality,
            'code_quality': self.EvaluateCodeQuality
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_metric = {
                executor.submit(func, Data): metric
                for metric, func in evaluation_tasks.items()
            }

            for future in concurrent.futures.as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    results[metric_name] = result
                except Exception as e:
                    logging.error(f"Failed to evaluate {metric_name}: {e}")
                    results[metric_name] = MetricResult(
                        metric_type=getattr(
                            MetricType,
                            metric_name.upper(),
                            MetricType.PERFORMANCE_CLAIMS
                        ),
                        value=0.0,
                        details={"error": str(e)},
                        latency_ms=0
                    )

        default_result = MetricResult(
            MetricType.PERFORMANCE_CLAIMS, 0.0, {}, 0
        )

        net_score_result = self.Evaluate_Net(results)
        model_name = getattr(Data, 'id', 'unknown-model')

        return {
            'name': model_name,
            'category': 'MODEL',
            'net_score': round(net_score_result.value, 2),
            'net_score_latency': net_score_result.latency_ms,
            'ramp_up_time': round(
                results.get('ramp_up_time', default_result).value, 2),
            'ramp_up_time_latency': results.get(
                'ramp_up_time', default_result).latency_ms,
            'bus_factor': round(
                results.get('bus_factor', default_result).value, 2),
            'bus_factor_latency': results.get(
                'bus_factor', default_result).latency_ms,
            'performance_claims': round(results.get(
                'performance_claims', default_result).value, 2),
            'performance_claims_latency': results.get(
                'performance_claims', default_result).latency_ms,
            'license': round(results.get('license', default_result).value, 2),
            'license_latency': results.get(
                'license', default_result).latency_ms,
            'size_score': self._extract_size_scores(
                results.get('size_score', default_result)),
            'size_score_latency': results.get(
                'size_score', default_result).latency_ms,
            'dataset_and_code_score': round(results.get(
                'dataset_and_code_score', default_result).value, 2),
            'dataset_and_code_score_latency': results.get(
                'dataset_and_code_score', default_result).latency_ms,
            'dataset_quality': round(results.get(
                'dataset_quality', default_result).value, 2),
            'dataset_quality_latency': results.get(
                'dataset_quality', default_result).latency_ms,
            'code_quality': round(
                results.get('code_quality', default_result).value, 2),
            'code_quality_latency': results.get(
                'code_quality', default_result).latency_ms
        }

    def Evaluate_Net(self, metric_results: Dict[str, MetricResult]
                     ) -> MetricResult:
        """Calculate weighted net score from individual metric results."""
        def _calculate_weighted_score(
                results: Dict[str, MetricResult]) -> Dict[str, Any]:
            # Define weights for each metric (must sum to 1.0)
            weights = {
                'performance_claims': 0.25,  # 25%
                'code_quality': 0.20,        # 20%
                'dataset_quality': 0.15,     # 15%
                'license': 0.15,             # 15%
                'bus_factor': 0.10,          # 10%
                'ramp_up_time': 0.10,        # 10%
                'size_score': 0.05           # 5%
            }

            weighted_sum = 0.0
            total_weight = 0.0
            metric_breakdown = {}

            for metric_name, weight in weights.items():
                if (metric_name in results and
                        isinstance(results[metric_name], MetricResult)):
                    metric_value = results[metric_name].value
                    contribution = metric_value * weight
                    weighted_sum += contribution
                    total_weight += weight
                    
                    metric_breakdown[metric_name] = {
                        "value": round(metric_value, 3),
                        "weight": weight,
                        "contribution": round(contribution, 3)
                    }
                else:
                    # Missing metric gets 0 score but still counts for weight
                    metric_breakdown[metric_name] = {
                        "value": 0.0,
                        "weight": weight,
                        "contribution": 0.0
                    }
                    total_weight += weight

            net_score = (weighted_sum / total_weight
                         if total_weight > 0 else 0.0)

            return max(0.0, min(1.0, net_score))

        try:
            start_time = time.time()
            calculation = _calculate_weighted_score(metric_results)
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            return MetricResult(
                metric_type=MetricType.NET_SCORE,
                value=calculation,
                details={
                    "mode": "weighted_sum",
                    "calculation_method": "deterministic",
                    "weights_used": {
                        'performance_claims': 0.25,
                        'code_quality': 0.20,
                        'dataset_quality': 0.15,
                        'license': 0.15,
                        'bus_factor': 0.10,
                        'ramp_up_time': 0.10,
                        'size_score': 0.05
                    },
                    "breakdown": calculation["metric_breakdown"],
                    "total_weight": calculation["total_weight"],
                    "weighted_sum": round(calculation["weighted_sum"], 3)
                },
                latency_ms=latency_ms,
            )

        except Exception as e:
            start_time = time.time()
            simple_average = (
                sum(r.value for r in metric_results.values()
                    if isinstance(r, MetricResult)) / len(metric_results)
                if metric_results else 0.0
            )
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            return MetricResult(
                metric_type=MetricType.NET_SCORE,
                value=max(0.0, min(1.0, simple_average)),
                details={
                    "mode": "simple_average_fallback",
                    "error": str(e)[:100],
                    "fallback_used": True
                },
                latency_ms=latency_ms
            )

    def EvaluatePerformanceClaims(self, Data: Model) -> MetricResult:
        def _compose_source_text(data: Model) -> str:
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

        def prepare_llm_prompt(data: Model) -> str:
            assert isinstance(data, Model)
            text = _compose_source_text(data)
            return (
                "OUTPUT FORMAT: JSON ONLY\n\n"
                "Analyze the model documentation and return this exact JSON "
                "format:\n\n"
                "{\n"
                '  "has_benchmark_datasets": true|false,\n'
                '  "has_quantitative_results": true|false,\n'
                '  "has_baseline_or_sota_comparison": true|false,\n'
                '  "notes": "Found GLUE scores and baseline comparisons"\n'
                "}\n\n"
                "Criteria:\n"
                "- has_benchmark_datasets: true if mentions datasets like "
                "GLUE, SQuAD, ImageNet, MMLU\n"
                "- has_quantitative_results: true if shows accuracy, F1, "
                "BLEU scores or metrics tables\n"
                "- has_baseline_or_sota_comparison: true if compares to "
                "other models/baselines\n\n"
                f"ANALYZE THIS TEXT:\n{text[:8000]}\n\n"
                "RESPOND WITH JSON ONLY:"
            )

        def parse_llm_response(response: str) -> Dict[str, Any]:
            try:
                if not response or not response.strip():
                    logging.warning("Empty LLM response received")
                    return {
                        "has_benchmark_datasets": False,
                        "has_quantitative_results": False,
                        "has_baseline_or_sota_comparison": False,
                        "notes": "Empty response from LLM"
                    }

                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()

                obj = json.loads(clean_response)
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
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse LLM response as JSON: {e}")
                logging.warning(f"Raw response: {response[:200]}...")
                return {
                    "has_benchmark_datasets": False,
                    "has_quantitative_results": False,
                    "has_baseline_or_sota_comparison": False,
                    "notes": f"JSON parse error: {str(e)[:100]}"
                }

        try:
            start_time = time.time()
            prompt = prepare_llm_prompt(Data)
            response = self.llm_manager.call_genai_api(prompt)

            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            parsed = parse_llm_response(response_text)

            score = 0.0
            if parsed["has_benchmark_datasets"]:
                score += 0.3
            if parsed["has_quantitative_results"]:
                score += 0.4
            if parsed["has_baseline_or_sota_comparison"]:
                score += 0.3
            if score > 1.0:
                score = 1.0

            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            details = {"mode": "llm", **parsed}

            return MetricResult(
                metric_type=MetricType.PERFORMANCE_CLAIMS,
                value=score,
                details=details,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            raise RuntimeError("LLM evaluation failed") from exc

    def EvaluateBusFactor(self, Data: Model) -> MetricResult:
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

        def _latest_commit_ts(data: Model) -> Optional[datetime]:
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

        def _contributors_count(data: Model) -> int:
            contribs = getattr(data, "repo_contributors", [])
            if not isinstance(contribs, list):
                return 0
            return sum(
                1 for c in contribs
                if int(c.get("contributions", 0)) > 0
            )

        try:
            start_time = time.time()

            n_contrib = _contributors_count(Data)
            last_ts = _latest_commit_ts(Data)

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

            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

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
                latency_ms=latency_ms,
            )

        except Exception as e:
            logging.error(f"Failed to evaluate bus factor: {e}")
            raise RuntimeError("Bus factor evaluation failed") from e

    def EvaluateSize(self, Data: Model) -> MetricResult:
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
            start_time = time.time()

            if isinstance(Data.repo_metadata, dict):
                s = (
                    Data.repo_metadata.get("size_mb")
                    or Data.repo_metadata.get("size")
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

                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)

                return MetricResult(
                    metric_type=MetricType.SIZE_SCORE,
                    value=sizeScore,
                    details={
                        "derived_size_mb": size_mb,
                        "raspberry_pi": r_pi,
                        "jetson_nano": j_nano,
                        "desktop_pc": d_pc,
                        "aws_server": aws
                    },
                    latency_ms=latency_ms,
                )
            else:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)

                logging.warning("Model repo_metadata is not a dictionary")
                return MetricResult(
                    metric_type=MetricType.SIZE_SCORE,
                    value=0.0,
                    details={"error": "repo_metadata is not a dictionary"},
                    latency_ms=latency_ms,
                )

        except Exception as e:
            logging.error(f"Failed to evaluate model size: {e}")
            raise RuntimeError("Size evaluation failed") from e

    def EvaluateDatasetAndCodeAvailabilityScore(self,
                                                Data: Model) -> MetricResult:
        def _compose_source_text(data: Model) -> str:
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

        def prepare_llm_prompt(data: Model) -> str:
            assert isinstance(data, Model)
            text = _compose_source_text(data)
            header = (
                "<|begin_of_text|><|start_header_id|>"
                "system<|end_header_id|>")

            return f"""{header}

                You are a model documentation analyzer. Check for dataset \
                and code references.
                Return ONLY valid JSON.

                <|eot_id|><|start_header_id|>user<|end_header_id|>

                Analyze this model documentation for dataset and code \
                availability:

                {text[:6000]}

                Check for:
                - lists_training_datasets: mentions specific dataset names
                - links_to_huggingface_datasets: has \
                huggingface.co/datasets/ URLs
                - links_to_code_repo: has GitHub/GitLab repository links

                Return ONLY:
                {{
                "lists_training_datasets": true,
                "links_to_huggingface_datasets": false,
                "links_to_code_repo": true,
                "notes": "Found dataset names and GitHub links"
                }}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """

        def parse_llm_response(response: str) -> Dict[str, Any]:
            obj = json.loads(response)
            return {
                "lists_training_datasets": bool(
                    obj.get("lists_training_datasets", False)
                ),
                "links_to_huggingface_datasets": bool(
                    obj.get("links_to_huggingface_datasets", False)
                ),
                "links_to_code_repo": bool(
                    obj.get("links_to_code_repo", False)
                ),
                "notes": str(obj.get("notes", ""))[:400],
            }

        try:
            start_time = time.time()
            prompt = prepare_llm_prompt(Data)
            response = self.llm_manager.call_genai_api(prompt)
            logging.info(f"LLM response content: {repr(response.content)}")
            parsed = parse_llm_response(response.content)

            score = 0.0
            if parsed["lists_training_datasets"]:
                score += 0.3
            if parsed["links_to_huggingface_datasets"]:
                score += 0.3
            if parsed["links_to_code_repo"]:
                score += 0.4
            if score > 1.0:
                score = 1.0

            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            details = {"mode": "llm", **parsed}

            return MetricResult(
                metric_type=MetricType.DATASET_AND_CODE_SCORE,
                value=score,
                details=details,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            raise RuntimeError("Dataset and code availability "
                               "evaluation failed") from exc

    def EvaluateCodeQuality(self, Data: Model) -> MetricResult:
        def _check_test_files(repo_contents: list) -> bool:
            if not isinstance(repo_contents, list):
                return False

            test_indicators = [
                'test', 'tests', 'testing', 'unittest', 'unit_test',
                'test_', '_test', 'spec', 'specs'
            ]

            for item in repo_contents:
                if isinstance(item, dict):
                    name = item.get('name', '').lower()
                    path = item.get('path', '').lower()

                    for indicator in test_indicators:
                        if (indicator in name or indicator in path or
                                name.startswith('test_') or
                                name.endswith('_test.py') or
                                name.endswith('_test') or
                                'test.py' in name):
                            return True
            return False

        def _check_dependency_management(repo_contents: list) -> bool:
            if not isinstance(repo_contents, list):
                return False

            dependency_files = [
                'requirements.txt', 'setup.py', 'pyproject.toml',
                'pipfile', 'poetry.lock', 'conda.yml', 'environment.yml'
            ]

            for item in repo_contents:
                if isinstance(item, dict):
                    name = item.get('name', '').lower()
                    if name in dependency_files:
                        return True
            return False

        def _analyze_code_with_llm(repo_contents: list) -> Dict[str, Any]:
            repo_summary = []
            for item in repo_contents[:50]:
                if isinstance(item, dict):
                    name = item.get('name', '')
                    item_type = item.get('type', '')
                    repo_summary.append(f"{item_type}: {name}")

            repo_text = "\n".join(repo_summary)

            prompt = (
                "CRITICAL: You MUST respond with ONLY valid JSON. "
                "No explanations, no markdown, no code blocks.\n\n"
                "Task: Analyze this repository structure for code quality. "
                "Return EXACTLY this JSON structure:\n\n"
                "{\n"
                '  "has_comprehensive_tests": true|false,\n'
                '  "shows_good_structure": true|false,\n'
                '  "has_documentation": true|false,\n'
                '  "notes": "analysis summary"\n'
                "}\n\n"
                "Rules:\n"
                "1. ONLY return JSON - nothing else\n"
                "2. Use true/false (lowercase) for booleans\n"
                "3. Keep notes under 30 characters\n\n"
                "Evaluation criteria:\n"
                "- has_comprehensive_tests: Are there test files covering "
                "multiple components?\n"
                "- shows_good_structure: Well-organized directories and "
                "separation of concerns?\n"
                "- has_documentation: README, docs, or documentation "
                "files present?\n\n"
                "Repository structure:\n"
                f"{repo_text}\n\n"
                "Remember: ONLY return the JSON object."
            )

            try:
                response = self.llm_manager.call_genai_api(prompt)
                logging.info(f"LLM response content: {repr(response.content)}")
                obj = json.loads(response.content)
                return {
                    "has_comprehensive_tests": bool(
                        obj.get("has_comprehensive_tests", False)
                    ),
                    "shows_good_structure": bool(
                        obj.get("shows_good_structure", False)
                    ),
                    "has_documentation": bool(
                        obj.get("has_documentation", False)
                    ),
                    "notes": str(obj.get("notes", ""))[:400],
                }
            except Exception:
                return {
                    "has_comprehensive_tests": False,
                    "shows_good_structure": False,
                    "has_documentation": False,
                    "notes": "LLM analysis failed"
                }

        try:
            start_time = time.time()
            repo_contents = getattr(Data, "repo_contents", [])

            if not isinstance(repo_contents, list):
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                return MetricResult(
                    metric_type=MetricType.CODE_QUALITY,
                    value=0.0,
                    details={"error": "No repository contents available"},
                    latency_ms=latency_ms,
                )

            has_tests = _check_test_files(repo_contents)
            has_dependency_mgmt = _check_dependency_management(repo_contents)

            llm_analysis = _analyze_code_with_llm(repo_contents)

            score = 0.0
            if has_tests:
                score += 0.4

            if llm_analysis["shows_good_structure"]:
                score += 0.3
            if has_dependency_mgmt:
                score += 0.3

            if score > 1.0:
                score = 1.0

            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            details = {
                "has_tests": has_tests,
                "has_dependency_management": has_dependency_mgmt,
                "lint_check_proxy": llm_analysis["shows_good_structure"],
                "llm_analysis": llm_analysis
            }

            return MetricResult(
                metric_type=MetricType.CODE_QUALITY,
                value=score,
                details=details,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logging.error(f"Failed to evaluate code quality: {e}")
            raise RuntimeError("Code quality evaluation failed") from e

    def EvaluateDatasetsQuality(self, Data: Model) -> MetricResult:

        def _compose_dataset_text(data: Model) -> str:
            dataset_texts = []

            dataset_cards = getattr(data, "dataset_cards", {})
            dataset_infos = getattr(data, "dataset_infos", {})

            for dataset_id, card in dataset_cards.items():
                card_text = ""
                if card is not None:
                    card_text += f"Dataset: {dataset_id}\n"
                    card_text += f"Card Data: {str(card)}\n"

                if dataset_id in dataset_infos:
                    info = dataset_infos[dataset_id]
                    card_text += f"Dataset Info: {str(info)}\n"

                if card_text.strip():
                    dataset_texts.append(card_text)

            combined_text = "\n\n".join(dataset_texts)
            if len(combined_text) > 16000:
                combined_text = combined_text[:16000] + "\n\n...[truncated]..."

            return combined_text

        def _prepare_dataset_llm_prompt(data: Model) -> str:
            dataset_text = _compose_dataset_text(data)

            if not dataset_text.strip():
                return ""

            return (
                "CRITICAL: You MUST respond with ONLY valid JSON. "
                "No explanations, no markdown, no code blocks.\n\n"
                "Task: Evaluate these dataset cards for quality indicators. "
                "Return EXACTLY this JSON structure:\n\n"
                "{\n"
                '  "has_comprehensive_card": true|false,\n'
                '  "has_clear_data_source": true|false,\n'
                '  "has_preprocessing_info": true|false,\n'
                '  "has_large_size": false|true,\n'
                '  "notes": "analysis summary"\n'
                "}\n\n"
                "Rules:\n"
                "1. ONLY return JSON - nothing else\n"
                "2. Use true/false (lowercase) for booleans\n"
                "3. Keep notes under 30 characters\n\n"
                "Evaluation criteria:\n"
                "- has_comprehensive_card: Complete dataset cards with "
                "description, usage, citation?\n"
                "- has_clear_data_source: Specific data sources mentioned?\n"
                "- has_preprocessing_info: Evidence of data processing, "
                "filtering, quality control?\n"
                "- has_large_size: Dataset appears large (>10k samples)?\n\n"
                "Dataset information:\n"
                f"{dataset_text}\n\n"
                "Remember: ONLY return the JSON object."
            )

        def _parse_dataset_llm_response(response: str) -> Dict[str, Any]:
            try:
                obj = json.loads(response)
                return {
                    "has_comprehensive_card": bool(
                        obj.get("has_comprehensive_card", False)
                    ),
                    "has_clear_data_source": bool(
                        obj.get("has_clear_data_source", False)
                    ),
                    "has_preprocessing_info": bool(
                        obj.get("has_preprocessing_info", False)
                    ),
                    "has_large_size": bool(
                        obj.get("has_large_size", False)
                    ),
                    "notes": str(obj.get("notes", ""))[:400],
                }
            except Exception:
                return {
                    "has_comprehensive_card": False,
                    "has_clear_data_source": False,
                    "has_preprocessing_info": False,
                    "has_large_size": False,
                    "notes": "Failed to parse LLM response"
                }

        try:
            start_time = time.time()
            dataset_cards = getattr(Data, "dataset_cards", {})
            dataset_infos = getattr(Data, "dataset_infos", {})

            if not dataset_cards and not dataset_infos:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                return MetricResult(
                    metric_type=MetricType.DATASET_QUALITY,
                    value=0.0,
                    details={"error": "No dataset information available"},
                    latency_ms=latency_ms,
                )

            prompt = _prepare_dataset_llm_prompt(Data)

            if not prompt:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                return MetricResult(
                    metric_type=MetricType.DATASET_QUALITY,
                    value=0.0,
                    details={"error": "No dataset content to analyze"},
                    latency_ms=latency_ms,
                )

            response = self.llm_manager.call_genai_api(prompt)
            logging.info(f"LLM response content: {repr(response.content)}")
            parsed = _parse_dataset_llm_response(response.content)

            score = 0.0
            if parsed["has_comprehensive_card"]:
                score += 0.4
            if parsed["has_clear_data_source"]:
                score += 0.2
            if parsed["has_preprocessing_info"]:
                score += 0.2
            if parsed["has_large_size"]:
                score += 0.2

            if score > 1.0:
                score = 1.0

            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            details = {
                "mode": "llm",
                "dataset_count": len(dataset_cards),
                **parsed
            }

            return MetricResult(
                metric_type=MetricType.DATASET_QUALITY,
                value=score,
                details=details,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logging.error(f"Failed to evaluate dataset quality: {e}")
            raise RuntimeError("Dataset quality evaluation failed") from e

    def EvaluateRampUpTime(self, Data: Model) -> MetricResult:
        def _compose_source_text(data: Model) -> str:
            readme = ""
            path = getattr(data, "readme_path", None)
            if path:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        readme = fh.read()
                except Exception:
                    readme = ""
            text = (readme).strip()
            if len(text) > 16000:
                text = text[:16000] + "\n\n...[truncated]..."
            return text

        def prepare_llm_prompt(data: Model) -> str:
            assert isinstance(data, Model)
            text = _compose_source_text(data)
            header = ("<|begin_of_text|><|start_header_id|>"
                      "system<|end_header_id|>")
            return f"""{header}

You are a README quality evaluator. Rate documentation quality.
Return ONLY valid JSON.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Rate this README for ramp-up time (0.0-0.5 each):

{text[:6000]}

Rate:
- quality_of_example_code: Code examples and usage instructions
- readme_coverage: Documentation completeness and structure

Return ONLY:
{{
  "quality_of_example_code": 0.4,
  "readme_coverage": 0.3,
  "notes": "Good examples, clear docs"
}}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        def parse_llm_response(response: str) -> Dict[str, Any]:
            if not response or not response.strip():
                raise ValueError("Empty response from LLM")

            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            obj = json.loads(response)
            quality_val = obj.get("quality_of_example_code", 0.0)
            if isinstance(quality_val, list) and quality_val:
                quality_val = quality_val[0]

            readme_val = obj.get("readme_coverage", 0.0)
            if isinstance(readme_val, list) and readme_val:
                readme_val = readme_val[0]

            return {
                "quality_of_example_code": float(quality_val),
                "readme_coverage": float(readme_val),
                "notes": str(obj.get("notes", ""))[:400],
            }

        try:
            start_time = time.time()
            prompt = prepare_llm_prompt(Data)
            response = self.llm_manager.call_genai_api(prompt)
            logging.info(f"LLM response content: {repr(response.content)}")
            parsed = parse_llm_response(response.content)

            score = 0.0
            score += parsed["quality_of_example_code"]
            score += parsed["readme_coverage"]

            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            details = {"mode": "llm", **parsed}

            return MetricResult(
                metric_type=MetricType.RAMP_UP_TIME,
                value=score,
                details=details,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            raise RuntimeError("LLM evaluation failed") from exc

    def EvaluateLicense(self, Data: Model) -> MetricResult:
        def _get_license_info(data: Model) -> str:
            """Extract license information from all available sources"""
            license_info = []

            card_obj = getattr(data, "card", None)
            if card_obj and isinstance(card_obj, dict):

                license_fields = ["license", "license_name", "license_link",
                                  "license_url"]
                for field in license_fields:
                    if field in card_obj and card_obj[field]:
                        license_info.append(f"{field}: {card_obj[field]}")

                description = card_obj.get("description", "")
                license_words = [
                    "license", "mit", "apache", "bsd", "gpl", "lgpl"
                ]

                if description and any(word in description.lower()
                                       for word in license_words):
                    license_info.append(f"description: {description}")

            repo_metadata = getattr(data, "repo_metadata", {})
            if isinstance(repo_metadata, dict):
                repo_license = repo_metadata.get("license")
                if repo_license:
                    if isinstance(repo_license, dict):
                        license_name = repo_license.get("name", "")
                        license_key = repo_license.get("key", "")
                        if license_name or license_key:
                            license_info.append(
                                f"repo_license: {license_name} "
                                f"({license_key})")
                    else:
                        license_info.append(f"repo_license: {repo_license}")

            return "\n".join(license_info) if license_info else ""

        def _classify_license(license_text: str) -> tuple:
            if not license_text:
                return 0.0,

            license_lower = license_text.lower()
            permissive_licenses = {
                "mit": "MIT License",
                "bsd": "BSD License",
                "bsd-2-clause": "BSD 2-Clause License",
                "bsd-3-clause": "BSD 3-Clause License",
                "apache": "Apache License",
                "apache-2.0": "Apache License 2.0",
                "apache 2.0": "Apache License 2.0",
                "lgpl-2.1": "LGPL v2.1",
                "lgpl v2.1": "LGPL v2.1",
                "lgpl-3.0": "LGPL v3.0",
                "lgpl v3.0": "LGPL v3.0",
                "isc": "ISC License",
                "unlicense": "Unlicense"
            }

            for license_key, license_name in permissive_licenses.items():
                if license_key in license_lower:
                    return (1.0, "rule_based",
                            f"Permissive license: {license_name}")

            restrictive_licenses = {
                "gpl-2.0": "GPL v2.0",
                "gpl v2.0": "GPL v2.0",
                "gpl-3.0": "GPL v3.0",
                "gpl v3.0": "GPL v3.0",
                "cc by-nc": "Creative Commons Non-Commercial",
                "cc-by-nc": "Creative Commons Non-Commercial",
                "non-commercial": "Non-Commercial License",
                "proprietary": "Proprietary License",
                "all rights reserved": "All Rights Reserved"
            }

            for license_key, license_name in restrictive_licenses.items():
                if license_key in license_lower:
                    return (0.0, "rule_based",
                            f"Restrictive license: {license_name}")

            license_keywords = ["license", "copyright", "terms", "conditions"]
            if any(keyword in license_lower for keyword in license_keywords):
                return (None, "llm_needed",
                        "Custom license requires LLM analysis")

            return 0.0, "rule_based", "Unclear or missing license information"

        def _prepare_llm_prompt(license_text: str) -> str:
            """Prepare LLM prompt for custom license analysis"""
            return (
                "OUTPUT FORMAT: JSON ONLY\n\n"
                "Analyze this license text for permissiveness. "
                "Return this JSON format:\n\n"
                "{\n"
                '  "permissiveness_score": 0.7,\n'
                '  "license_type": "Custom permissive",\n'
                '  "allows_commercial": true,\n'
                '  "allows_modification": true,\n'
                '  "notes": "Allows commercial use with attribution"\n'
                "}\n\n"
                "Scoring rules (STRICT):\n"
                "- 1.0: MIT/Apache/BSD-like (very permissive)\n"
                "- 0.8-0.9: Permissive with minor restrictions\n"
                "- 0.5-0.7: Some commercial/modification limits\n"
                "- 0.1-0.4: Significant restrictions\n"
                "- 0.0: GPL/Non-commercial/Highly restrictive\n\n"
                f"LICENSE TEXT:\n{license_text[:2000]}\n\n"
                "RESPOND WITH JSON ONLY:"
            )

        def _parse_llm_response(response: str) -> Dict[str, Any]:
            """Parse LLM response for license analysis"""
            try:
                obj = json.loads(response)
                return {
                    "permissiveness_score": float(
                        obj.get("permissiveness_score", 0.0)),
                    "license_type": str(obj.get("license_type", "Unknown")),
                    "allows_commercial": bool(
                        obj.get("allows_commercial", False)),
                    "allows_modification": bool(
                        obj.get("allows_modification", False)),
                    "notes": str(obj.get("notes", ""))[:200],
                }
            except Exception:
                return {
                    "permissiveness_score": 0.0,
                    "license_type": "Parse error",
                    "allows_commercial": False,
                    "allows_modification": False,
                    "notes": "Failed to parse LLM response"
                }

        try:
            start_time = time.time()
            license_text = _get_license_info(Data)

            score, classification_type, reason = _classify_license(
                license_text)

            if score is not None:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                details = {
                    "classification_method": classification_type,
                    "license_text": license_text[:500] if license_text else "",
                    "reason": reason,
                }

                return MetricResult(
                    metric_type=MetricType.LICENSE,
                    value=score,
                    details=details,
                    latency_ms=latency_ms,
                )

            else:
                if not license_text:
                    end_time = time.time()
                    latency_ms = int((end_time - start_time) * 1000)
                    return MetricResult(
                        metric_type=MetricType.LICENSE,
                        value=0.0,
                        details={"error": "No license information available"},
                        latency_ms=latency_ms,
                    )

                prompt = _prepare_llm_prompt(license_text)
                response = self.llm_manager.call_genai_api(prompt)
                parsed = _parse_llm_response(response.content)
                llm_score = max(0.0, min(1.0, parsed["permissiveness_score"]))

                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)

                details = {
                    "classification_method": "llm_analysis",
                    "license_text": license_text[:500],
                    "llm_analysis": parsed,
                }

                return MetricResult(
                    metric_type=MetricType.LICENSE,
                    value=llm_score,
                    details=details,
                    latency_ms=latency_ms,
                )

        except Exception as e:
            logging.error(f"Failed to evaluate license: {e}")
            raise RuntimeError("License evaluation failed") from e
