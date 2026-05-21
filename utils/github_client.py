import os
import re
import requests
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse

class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        """Initializes the GitHub REST API Client."""
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def parse_repo_url(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Parses a standard GitHub URL to extract (owner, repo).
        e.g., https://github.com/google/guava -> ('google', 'guava')
        """
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1].replace(".git", "")
                return owner, repo
        except Exception:
            pass
        return None

    def get_repo_details(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetches repository metadata details from GitHub."""
        url = f"https://api.github.com/repos/{owner}/{str(repo)}"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_default_branch(self, owner: str, repo: str) -> str:
        """Retrieves the default branch (e.g., 'main' or 'master') for a repository."""
        try:
            details = self.get_repo_details(owner, repo)
            return details.get("default_branch", "main")
        except Exception:
            return "main"

    def download_repo_zip(self, owner: str, repo: str, branch: str, target_path: str) -> bool:
        """
        Downloads the entire repository zipball for a branch.
        Saves it to the specified target path.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
        response = requests.get(url, headers=self.headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False

    def fetch_pull_requests(self, owner: str, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """Fetches active pull requests for the repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": 20}
        response = requests.get(url, headers=self.headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []

    def fetch_pr_details(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Fetches complete details for a single pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def fetch_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """
        Fetches the unified diff of a pull request.
        Uses the media type parameter header.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        diff_headers = self.headers.copy()
        diff_headers["Accept"] = "application/vnd.github.v3.diff"
        
        response = requests.get(url, headers=diff_headers, timeout=15)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch PR diff: HTTP {response.status_code}")

    def fetch_commits(self, owner: str, repo: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches recent commits on the default branch."""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"per_page": limit}
        response = requests.get(url, headers=self.headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
