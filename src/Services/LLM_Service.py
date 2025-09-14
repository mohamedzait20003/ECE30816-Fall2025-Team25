import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests


@dataclass
class LLMResponse:
    """Response data class for LLM API calls."""
    content: str
    usage_stats: Optional[Dict] = None
    model_used: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class PromptTemplate:
    """Template for structuring prompts."""
    system_prompt: str
    user_prompt_template: str
    context_variables: Optional[Dict[str, str]] = None


class LLMService:
    """Service for interacting with the Gemini API."""
    
    def __init__(self):
        """Initialize the LLM service with API configuration."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logging.warning(
                "GEMINI_API_KEY not found in environment variables. "
                "LLM service will not be functional."
            )
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        # You can change this to gemini-1.5-pro for more complex tasks
        self.default_model = "gemini-1.5-flash"
        self.max_tokens = 8192
        self.temperature = 0.7
        
        # Initialize prompt templates
        self._init_prompt_templates()
    
    def _init_prompt_templates(self):
        """Initialize common prompt templates."""
        self.prompt_templates = {
            "model_analysis": PromptTemplate(
                system_prompt=(
                    "You are an expert AI researcher and model analyst. "
                    "Your task is to analyze machine learning models and "
                    "provide comprehensive insights based on the provided "
                    "metadata and documentation."
                ),
                user_prompt_template="""
                Analyze the following machine learning model:

                **Model Information:**
                - Name: {model_name}
                - ID: {model_id}
                - Description: {model_description}

                **Model Card Data:**
                {model_card}

                **README Content:**
                {readme_content}

                **Associated Datasets:**
                {datasets_info}

                **Repository Information:**
                {repo_info}

                Please provide a comprehensive analysis including:
                1. Model purpose and capabilities
                2. Training methodology and datasets used
                3. Performance characteristics
                4. Potential use cases and applications
                5. Limitations and considerations
                6. Technical specifications
                7. Overall assessment and recommendations

                Format your response in clear sections with appropriate
                headings.
                """
                ),
            
            "dataset_summary": PromptTemplate(
                system_prompt=(
                    "You are a data science expert specializing in dataset "
                    "analysis and curation. Provide clear, concise summaries "
                    "of datasets."
                ),
                user_prompt_template="""
                Summarize the following dataset:

                **Dataset Name:** {dataset_name}
                **Dataset ID:** {dataset_id}

                **Dataset Card:**
                {dataset_card}

                **Dataset Info:**
                {dataset_info}

                Provide a summary that includes:
                1. Dataset purpose and domain
                2. Data structure and format
                3. Size and scope
                4. Key features and characteristics
                5. Potential applications
                6. Quality and reliability notes

                Keep the summary concise but informative.
                """
                ),
            
            "code_review": PromptTemplate(
                system_prompt=(
                    "You are a senior software engineer with expertise in "
                    "machine learning codebases. Analyze code repositories "
                    "and provide constructive feedback."
                ),
                user_prompt_template="""
                Review the following code repository:

                **Repository:** {repo_name}
                **Description:** {repo_description}

                **Repository Structure:**
                {repo_contents}

                **Contributors:** {contributors}

                **Recent Commits:**
                {recent_commits}

                **Additional Metadata:**
                {repo_metadata}

                Provide a code review that covers:
                1. Code organization and structure
                2. Documentation quality
                3. Project maturity and maintenance
                4. Community engagement
                5. Technical implementation notes
                6. Suggestions for improvement

                Focus on constructive feedback and actionable insights.
                """
            )
        }
    
    def prepare_prompt(self, template_name: str, **kwargs) -> str:
        """
        Prepare a prompt using a predefined template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.prompt_templates:
            raise ValueError(
                f"Template '{template_name}' not found. Available "
                f"templates: {list(self.prompt_templates.keys())}"
            )
        
        template = self.prompt_templates[template_name]
        
        try:
            # Format the user prompt with provided variables
            user_prompt = template.user_prompt_template.format(**kwargs)
            
            # Combine system and user prompts
            full_prompt = f"{template.system_prompt}\n\n{user_prompt}"
            
            return full_prompt
            
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(
                f"Missing required variable '{missing_key}' for "
                f"template '{template_name}'"
            )
    
    def prepare_custom_prompt(self, system_prompt: str, user_prompt: str,
                              **context) -> str:
        """
        Prepare a custom prompt with system and user components.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query or task description
            **context: Additional context variables to format into prompts
            
        Returns:
            Formatted prompt string
        """
        try:
            if context:
                formatted_user_prompt = user_prompt.format(**context)
            else:
                formatted_user_prompt = user_prompt
            return f"{system_prompt}\n\n{formatted_user_prompt}"
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(
                f"Missing context variable '{missing_key}' for custom prompt"
            )
    
    def call_gemini_api(self, prompt: str, model: Optional[str] = None,
                        **generation_config) -> LLMResponse:
        """
        Make a direct API call to Gemini using REST API.
        
        Args:
            prompt: The formatted prompt to send
            model: Model to use (defaults to self.default_model)
            **generation_config: Additional generation parameters
            
        Returns:
            LLMResponse object with the API response
        """
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. Cannot make API calls."
            )
        
        model_name = model or self.default_model
        url = f"{self.base_url}/models/{model_name}:generateContent"
        
        # Prepare request payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": generation_config.get(
                    "temperature", self.temperature
                ),
                "maxOutputTokens": generation_config.get(
                    "max_tokens", self.max_tokens
                ),
                "topP": generation_config.get("top_p", 0.95),
                "topK": generation_config.get("top_k", 40)
            }
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        try:
            response = requests.post(
                url, json=payload, headers=headers, params=params
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract content from response
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise ValueError(
                    "No response candidates received from Gemini API"
                )
            
            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                raise ValueError("No content parts in API response")
            
            content = content_parts[0].get("text", "")
            
            # Extract usage statistics if available
            usage_stats = response_data.get("usageMetadata", {})
            finish_reason = candidates[0].get("finishReason", "STOP")
            
            return LLMResponse(
                content=content,
                usage_stats=usage_stats,
                model_used=model_name,
                finish_reason=finish_reason
            )
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise RuntimeError(f"Failed to call Gemini API: {e}")
        except (KeyError, IndexError) as e:
            logging.error(f"Failed to parse API response: {e}")
            raise RuntimeError(f"Invalid API response format: {e}")
    
    def analyze_model(self, model_data) -> LLMResponse:
        """
        Analyze a model using the model_analysis template.
        
        Args:
            model_data: ModelData object with model information
            
        Returns:
            LLMResponse with the analysis
        """
        # Prepare context variables
        context = {
            "model_name": getattr(model_data, 'id', 'Unknown'),
            "model_id": getattr(model_data, 'id', 'Unknown'),
            "model_description": str(getattr(
                model_data, 'card', 'No description available'
            )),
            "model_card": str(getattr(
                model_data, 'card', 'No model card available'
            )),
            "readme_content": self._get_readme_content(model_data),
            "datasets_info": self._format_datasets_info(model_data),
            "repo_info": self._format_repo_info(model_data)
        }
        
        # Prepare and send prompt
        prompt = self.prepare_prompt("model_analysis", **context)
        return self.call_gemini_api(prompt)
    
    def summarize_dataset(self, dataset_id: str, dataset_info,
                          dataset_card) -> LLMResponse:
        """
        Summarize a dataset using the dataset_summary template.
        
        Args:
            dataset_id: Dataset identifier
            dataset_info: Dataset information object
            dataset_card: Dataset card data
            
        Returns:
            LLMResponse with the summary
        """
        context = {
            "dataset_name": dataset_id,
            "dataset_id": dataset_id,
            "dataset_card": (
                str(dataset_card) if dataset_card
                else "No dataset card available"
            ),
            "dataset_info": (
                str(dataset_info) if dataset_info
                else "No dataset info available"
            )
        }
        
        prompt = self.prepare_prompt("dataset_summary", **context)
        return self.call_gemini_api(prompt)
    
    def review_code_repository(self, model_data) -> LLMResponse:
        """
        Review a code repository using the code_review template.
        
        Args:
            model_data: ModelData object with repository information
            
        Returns:
            LLMResponse with the code review
        """
        repo_metadata = getattr(model_data, 'repo_metadata', {})
        
        context = {
            "repo_name": repo_metadata.get('full_name', 'Unknown Repository'),
            "repo_description": repo_metadata.get(
                'description', 'No description available'
            ),
            "repo_contents": self._format_repo_contents(model_data),
            "contributors": self._format_contributors(model_data),
            "recent_commits": self._format_commits(model_data),
            "repo_metadata": (
                json.dumps(repo_metadata, indent=2) if repo_metadata
                else "No metadata available"
            )
        }
        
        prompt = self.prepare_prompt("code_review", **context)
        return self.call_gemini_api(prompt)
    
    def _get_readme_content(self, model_data) -> str:
        """Extract README content from model data."""
        readme_path = getattr(model_data, 'readme_path', None)
        if readme_path:
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 2000:
                        return content[:2000] + "..."
                    else:
                        return content
            except Exception as e:
                logging.warning(f"Failed to read README file: {e}")
        return "No README content available"
    
    def _format_datasets_info(self, model_data) -> str:
        """Format dataset information for prompts."""
        dataset_ids = getattr(model_data, 'dataset_ids', [])
        if not dataset_ids:
            return "No associated datasets"
        
        return "\n".join([f"- {dataset_id}" for dataset_id in dataset_ids])
    
    def _format_repo_info(self, model_data) -> str:
        """Format repository information for prompts."""
        repo_metadata = getattr(model_data, 'repo_metadata', {})
        if not repo_metadata:
            return "No repository information available"
        
        info = []
        info.append(f"Repository: {repo_metadata.get('full_name', 'Unknown')}")
        info.append(
            f"Description: "
            f"{repo_metadata.get('description', 'No description')}"
        )
        info.append(f"Stars: {repo_metadata.get('stargazers_count', 0)}")
        info.append(f"Forks: {repo_metadata.get('forks_count', 0)}")
        info.append(f"Language: {repo_metadata.get('language', 'Unknown')}")
        info.append(f"Created: {repo_metadata.get('created_at', 'Unknown')}")
        info.append(
            f"Last Updated: {repo_metadata.get('updated_at', 'Unknown')}"
        )
        
        return "\n".join(info)
    
    def _format_repo_contents(self, model_data) -> str:
        """Format repository contents for prompts."""
        contents = getattr(model_data, 'repo_contents', [])
        if not contents:
            return "No repository contents available"
        
        formatted = []
        for item in contents[:20]:  # Limit to first 20 items
            item_type = item.get('type', 'unknown')
            item_name = item.get('name', 'unknown')
            formatted.append(f"- {item_name} ({item_type})")
        
        if len(contents) > 20:
            formatted.append(f"... and {len(contents) - 20} more items")
        
        return "\n".join(formatted)
    
    def _format_contributors(self, model_data) -> str:
        """Format contributors information for prompts."""
        contributors = getattr(model_data, 'repo_contributors', [])
        if not contributors:
            return "No contributors information available"
        
        formatted = []
        for contributor in contributors[:10]:  # Limit to top 10 contributors
            login = contributor.get('login', 'unknown')
            contributions = contributor.get('contributions', 0)
            formatted.append(f"- {login} ({contributions} contributions)")
        
        return "\n".join(formatted)
    
    def _format_commits(self, model_data) -> str:
        """Format recent commits for prompts."""
        commits = getattr(model_data, 'repo_commit_history', [])
        if not commits:
            return "No commit history available"
        
        formatted = []
        for commit in commits[:5]:  # Limit to 5 most recent commits
            commit_info = commit.get('commit', {})
            # First line only
            message = commit_info.get('message', '').split('\n')[0]
            author_info = commit_info.get('author', {})
            author = author_info.get('name', 'Unknown')
            date = author_info.get('date', 'Unknown')
            
            formatted.append(f"- {message[:100]} by {author} on {date}")
        
        return "\n".join(formatted)
    
    def add_custom_template(self, name: str, system_prompt: str,
                            user_prompt_template: str) -> None:
        """
        Add a custom prompt template.
        
        Args:
            name: Name for the template
            system_prompt: System-level instructions
            user_prompt_template: Template string with format placeholders
        """
        self.prompt_templates[name] = PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template
        )
    
    def list_available_templates(self) -> List[str]:
        """List all available prompt templates."""
        return list(self.prompt_templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, str]:
        """Get information about a specific template."""
        if template_name not in self.prompt_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.prompt_templates[template_name]
        return {
            "name": template_name,
            "system_prompt": template.system_prompt,
            "user_prompt_template": template.user_prompt_template
        }
