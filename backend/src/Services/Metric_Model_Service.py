import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from Models import Model
from lib.LLM_Manager import LLMManager
from lib.Metric_Result import MetricResult, MetricType
from Helpers import _parse_iso8601, _months_between


class ModelMetricService:
    def __init__(self) -> None:
        self.llm_manager = LLMManager()

    def EvaluateModel(
        self, model_description: str, dataset_description: str
    ) -> MetricResult:
        return MetricResult(
            metric_type=MetricType.PERFORMANCE_CLAIMS,
            value=0.0,
            details={
                "info": "Model evaluation not yet implemented"
            },
            latency_ms=0,
            error=None
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
            prompt = prepare_llm_prompt(Data)
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

    def EvaluateDatasetAndCodeScore(self, Data: Model) -> MetricResult:
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
                "You are evaluating a machine learning model card/README "
                "for dataset and code references. "
                "Only use the provided text. Return STRICT JSON with these "
                "boolean fields and a short notes string:\n"
                "{\n"
                '  "lists_training_datasets": true|false,\n'
                '  "links_to_huggingface_datasets": true|false,\n'
                '  "links_to_code_repo": true|false,\n'
                '  "notes": "brief rationale"\n'
                "}\n\n"
                "Definitions:\n"
                "- lists_training_datasets: README explicitly mentions "
                "training datasets used (dataset names, descriptions).\n"
                "- links_to_huggingface_datasets: Contains links to "
                "Hugging Face Hub datasets (huggingface.co/datasets/).\n"
                "- links_to_code_repo: Contains links to training/"
                "fine-tuning code repositories (GitHub, GitLab, etc.).\n\n"
                "=== BEGIN TEXT ===\n"
                f"{text}\n"
                "=== END TEXT ===\n"
            )

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
            prompt = prepare_llm_prompt(Data)
            response = self.llm_manager.call_gemini_api(prompt)
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

            details = {"mode": "llm", **parsed}

            return MetricResult(
                metric_type=MetricType.DATASET_AND_CODE_SCORE,
                value=score,
                details=details,
                latency_ms=0,
            )

        except Exception as exc:
            raise RuntimeError("Dataset and code evaluation failed") from exc

    def EvaluateAvailability(self, Model: Model) -> MetricResult:
        try:
            if isinstance(Model.repo_metadata, dict):
                is_private = Model.repo_metadata.get("private", False)
                if isinstance(is_private, str):
                    is_private = is_private.lower() == "true"
                availability = 0.0 if is_private else 1.0
                details = {"is_private": is_private}
            else:
                availability = 0.0
                details = {"error": "repo_metadata is not a dictionary"}

            return MetricResult(
                metric_type=MetricType.AVAILABILITY,
                value=availability,
                details=details,
                latency_ms=0,
            )

        except Exception as e:
            logging.error(f"Failed to evaluate availability: {e}")
            raise RuntimeError("Availability evaluation failed") from e

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
                "You are analyzing a code repository structure for quality. "
                "Based on the file/directory listing below, evaluate "
                "code quality indicators. "
                "Return STRICT JSON with these boolean fields:\n"
                "{\n"
                '  "has_comprehensive_tests": true|false,\n'
                '  "shows_good_structure": true|false,\n'
                '  "has_documentation": true|false,\n'
                '  "notes": "brief analysis"\n'
                "}\n\n"
                "Definitions:\n"
                "- has_comprehensive_tests: Tests appear to cover "
                "multiple components/modules\n"
                "- shows_good_structure: Clear separation of concerns, "
                "organized directories\n"
                "- has_documentation: README, docs, or inline "
                "documentation present\n\n"
                "=== REPOSITORY STRUCTURE ===\n"
                f"{repo_text}\n"
                "=== END STRUCTURE ===\n"
            )

            try:
                response = self.llm_manager.call_gemini_api(prompt)
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
            repo_contents = getattr(Data, "repo_contents", [])

            if not isinstance(repo_contents, list):
                return MetricResult(
                    metric_type=MetricType.CODE_QUALITY,
                    value=0.0,
                    details={"error": "No repository contents available"},
                    latency_ms=0,
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
                latency_ms=0,
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
                "You are evaluating machine learning dataset cards "
                "for quality. "
                "Based on the dataset information provided, evaluate quality "
                "indicators. Return STRICT JSON with these boolean fields "
                "and a notes string:\n"
                "{\n"
                '  "has_comprehensive_card": true|false,\n'
                '  "has_clear_data_source": true|false,\n'
                '  "has_preprocessing_info": true|false,\n'
                '  "has_large_size": true|false,\n'
                '  "notes": "brief analysis"\n'
                "}\n\n"
                "Definitions:\n"
                "- has_comprehensive_card: Contains sections like "
                "Description, Citation, Licensing, Usage\n"
                "- has_clear_data_source: Mentions specific data sources "
                "(Wikipedia, Common Crawl, etc.)\n"
                "- has_preprocessing_info: Evidence of data preprocessing, "
                "splits, quality controls, filtering\n"
                "- has_large_size: Dataset size > 10k entries/samples "
                "(look for numbers, size indicators)\n\n"
                "=== DATASET INFORMATION ===\n"
                f"{dataset_text}\n"
                "=== END DATASET INFO ===\n"
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
            dataset_cards = getattr(Data, "dataset_cards", {})
            dataset_infos = getattr(Data, "dataset_infos", {})

            if not dataset_cards and not dataset_infos:
                return MetricResult(
                    metric_type=MetricType.DATASET_QUALITY,
                    value=0.0,
                    details={"error": "No dataset information available"},
                    latency_ms=0,
                )

            prompt = _prepare_dataset_llm_prompt(Data)

            if not prompt:
                return MetricResult(
                    metric_type=MetricType.DATASET_QUALITY,
                    value=0.0,
                    details={"error": "No dataset content to analyze"},
                    latency_ms=0,
                )

            response = self.llm_manager.call_gemini_api(prompt)
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

            details = {
                "mode": "llm",
                "dataset_count": len(dataset_cards),
                **parsed
            }

            return MetricResult(
                metric_type=MetricType.DATASET_QUALITY,
                value=score,
                details=details,
                latency_ms=0,
            )

        except Exception as e:
            logging.error(f"Failed to evaluate dataset quality: {e}")
            raise RuntimeError("Dataset quality evaluation failed") from e

    def EvaluateRampUpTime(self, Model: ModelManager) -> MetricResult:
        def _compose_source_text(data: ModelManager) -> str:
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

        def prepare_llm_prompt(data: ModelManager) -> str:
            assert isinstance(data, ModelManager)
            text = _compose_source_text(data)
            return (
                "You are evaluating a machine learning model card/README. "
                "Only use the provided text. Return STRICT JSON with these "
                "float fields and a short notes string:\n"
                "{\n"
                '  "quality_of_example_code": 0.3,\n'
                '  "readme_coverage": 0.4,\n'
                '  "notes": "brief rationale"\n'
                "}\n\n"
                "Definitions:\n"
                "- Quality of example code: How comprehensive and"
                " well-documented are the examples provided in the README?"
                " (0.0 = none, 0.5 = excellent). Return a single float value.\n"
                "- Readme coverage: How detailed and clear is the README?"
                " (Contains headings like 'Usage', 'Training Data', "
                "'Evaluation', etc.)."
                " (0.0 = none, 0.5 = excellent). Return a single float value.\n"
                "=== BEGIN TEXT ===\n"
                f"{text}\n"
                "=== END TEXT ===\n"
            )

        def parse_llm_response(response: str) -> Dict[str, Any]:
            logging.info(f"LLM Response received: {repr(response)}")
            if not response or not response.strip():
                raise ValueError("Empty response from LLM")
            
            # Remove markdown code block markers if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            if response.startswith("```"):
                response = response[3:]   # Remove ```
            if response.endswith("```"):
                response = response[:-3]  # Remove trailing ```
            response = response.strip()
            
            obj = json.loads(response)  # let it raise if bad JSON
            
            # Handle array values - take the first value if it's an array
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
            prompt = prepare_llm_prompt(Model)
            logging.info(f"Calling LLM with prompt length: {len(prompt)}")
            response = self.llm_manager.call_gemini_api(prompt)
            logging.info(f"LLM response object: {response}")
            logging.info(f"LLM response content: {repr(response.content)}")
            parsed = parse_llm_response(response.content)

            score = 0.0
            score += parsed["quality_of_example_code"]
            score += parsed["readme_coverage"]

            details = {"mode": "llm", **parsed}

            return MetricResult(
                metric_type=MetricType.RAMP_UP_TIME,
                value=score,
                details=details,
                latency_ms=0,
            )

        except Exception as exc:
            raise RuntimeError("LLM evaluation failed") from exc