import re
import requests
from typing import Dict, List, Tuple, Union, Optional


class GitHubAPIManager:
    GITHUB_API_BASE = "https://api.github.com"

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize GitHub API Manager with optional token."""
        self.token = token
        if token:
            self.headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }
        else:
            self.headers = {
                "Accept": "application/vnd.github.v3+json",
            }

    @staticmethod
    def code_link_to_repo(code_link: str) -> Tuple[str, str]:
        """Converts a code repository link to a repo identifier."""
        match = re.search(r"github\.com/([^/]+)/([^/]+)", code_link)
        if not match:
            raise ValueError(f"Invalid GitHub repo URL: {code_link}")
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo

    def github_request(
        self,
        path: str,
        params: Optional[Dict] = None
    ) -> Union[Dict, List]:
        """Make a request to GitHub API."""
        if not self.token:
            raise ValueError("GitHub token is required for API requests")

        url = f"{self.GITHUB_API_BASE}{path}"
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code != 200:
            raise ValueError(
                f"GitHub API request failed: {response.status_code} "
                f"{response.text}"
            )
        return response.json()

    def get_repo_contents(
        self,
        owner: str,
        repo: str,
        path: str = ""
    ) -> List:
        """Retrieves the contents of a GitHub repository."""
        if path:
            path = f"/{path.lstrip('/')}"
        req = self.github_request(
            path=f"/repos/{owner}/{repo}/contents{path}"
        )
        assert isinstance(req, list), "Expected a list of contents"
        return req

    def get_repo_info(self, owner: str, repo: str) -> Dict:
        """Get repository information."""
        result = self.github_request(path=f"/repos/{owner}/{repo}")
        assert isinstance(result, dict), "Expected a dict for repo info"
        return result

    def get_repo_readme(self, owner: str, repo: str) -> Dict:
        """Get repository README."""
        result = self.github_request(path=f"/repos/{owner}/{repo}/readme")
        assert isinstance(result, dict), "Expected a dict for README"
        return result
