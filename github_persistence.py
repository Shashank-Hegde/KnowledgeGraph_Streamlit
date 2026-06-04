"""
GitHub Persistence Layer
========================
Pushes and pulls calibrated_weights.json and feedback_log.json
to/from a GitHub repo via the GitHub API.

This allows:
  - Multiple doctors to share the same calibrated weights
  - Weight updates from Streamlit Cloud to persist across sessions
  - Full audit trail in git history (every weight change is a commit)

Requires: GitHub Personal Access Token with 'repo' scope
stored in Streamlit secrets.
"""

import json
import base64
import requests
from typing import Optional


class GitHubPersistence:
    """Push/pull JSON files to a GitHub repo."""

    def __init__(self, token, repo, branch="main"):
        """
        Args:
            token: GitHub Personal Access Token (with repo scope)
            repo: "owner/repo-name" (e.g., "ohealth/triage-engine")
            branch: branch to read/write (default: "main")
        """
        self.token = token
        self.repo = repo
        self.branch = branch
        self.base_url = "https://api.github.com/repos/%s" % repo
        self.headers = {
            "Authorization": "token %s" % token,
            "Accept": "application/vnd.github.v3+json",
        }

    def pull_file(self, filepath):
        """
        Pull a file from the GitHub repo.
        Returns the parsed JSON content, or None if not found.
        """
        url = "%s/contents/%s?ref=%s" % (self.base_url, filepath, self.branch)
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 404:
                return None, None  # file doesn't exist yet
            resp.raise_for_status()
            data = resp.json()
            content = base64.b64decode(data["content"]).decode("utf-8")
            sha = data["sha"]
            return json.loads(content), sha
        except Exception as e:
            print("GitHub pull error: %s" % str(e))
            return None, None

    def push_file(self, filepath, content_dict, message, sha=None):
        """
        Push a JSON file to the GitHub repo.

        Args:
            filepath: path in repo (e.g., "calibrated_weights.json")
            content_dict: dict to serialize as JSON
            message: commit message
            sha: SHA of existing file (for updates). If None, creates new file.

        Returns: True if successful, False otherwise.
        """
        url = "%s/contents/%s" % (self.base_url, filepath)
        content_b64 = base64.b64encode(
            json.dumps(content_dict, indent=2).encode("utf-8")
        ).decode("utf-8")

        payload = {
            "message": message,
            "content": content_b64,
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha

        try:
            resp = requests.put(url, headers=self.headers, json=payload, timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            print("GitHub push error: %s" % str(e))
            return False

    def pull_weights(self):
        """Pull calibrated_weights.json from GitHub."""
        return self.pull_file("calibrated_weights.json")

    def push_weights(self, weights_dict, sha=None, session_info=""):
        """Push updated weights to GitHub."""
        msg = "Update weights from doctor review"
        if session_info:
            msg += " (%s)" % session_info
        return self.push_file("calibrated_weights.json", weights_dict, msg, sha=sha)

    def pull_feedback_log(self):
        """Pull feedback_log.json from GitHub."""
        return self.pull_file("feedback_log.json")

    def push_feedback_log(self, log_list, sha=None):
        """Push updated feedback log to GitHub."""
        return self.push_file(
            "feedback_log.json", log_list,
            "Add feedback entry (%d total)" % len(log_list),
            sha=sha
        )

    def is_configured(self):
        """Check if GitHub credentials are valid."""
        try:
            resp = requests.get(
                self.base_url, headers=self.headers, timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False
