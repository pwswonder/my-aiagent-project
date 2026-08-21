from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

REPO_PATH_RE = re.compile(r"^/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


@dataclass(slots=True)
class OfficialCodeSource:
    url: str
    full_name: str
    commit_sha: str
    license_spdx: str | None
    description: str | None
    verified: bool
    reference_files: list[str]
    reference_excerpts: dict[str, str]


class GitHubSourceResolver:
    def __init__(self, token: str | None = None, timeout: float = 8.0) -> None:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)

    @staticmethod
    def full_name(url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            return None
        match = REPO_PATH_RE.match(parsed.path)
        return "/".join(match.groups()) if match else None

    def resolve(self, url: str, paper_title: str) -> OfficialCodeSource | None:
        full_name = self.full_name(url)
        if not full_name:
            return None
        response = self.client.get(f"https://api.github.com/repos/{full_name}")
        if response.status_code != 200:
            return None
        payload = response.json()
        default_branch = payload.get("default_branch") or "main"
        commit_response = self.client.get(f"https://api.github.com/repos/{full_name}/commits/{default_branch}")
        if commit_response.status_code != 200:
            return None
        description = payload.get("description")
        haystack = f"{full_name} {description or ''}".lower()
        title_tokens = {token for token in re.findall(r"[a-z0-9]+", paper_title.lower()) if len(token) > 4}
        verified = bool(title_tokens and title_tokens.intersection(re.findall(r"[a-z0-9]+", haystack)))
        license_payload = payload.get("license") or {}
        reference_files: list[str] = []
        tree_response = self.client.get(
            f"https://api.github.com/repos/{full_name}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if tree_response.status_code == 200:
            candidates = [
                str(item.get("path"))
                for item in tree_response.json().get("tree", [])
                if item.get("type") == "blob"
                and str(item.get("path", "")).lower().endswith((".py", ".yaml", ".yml", ".json"))
                and any(
                    token in str(item.get("path", "")).lower()
                    for token in ("model", "network", "module", "config", "train")
                )
            ]
            reference_files = sorted(candidates)[:50]
        reference_excerpts: dict[str, str] = {}
        for path in reference_files[:5]:
            content_response = self.client.get(
                f"https://api.github.com/repos/{full_name}/contents/{path}",
                params={"ref": commit_response.json()["sha"]},
            )
            if content_response.status_code != 200:
                continue
            encoded = content_response.json().get("content")
            if encoded:
                decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
                reference_excerpts[path] = decoded[:20_000]
        return OfficialCodeSource(
            url=f"https://github.com/{full_name}",
            full_name=full_name,
            commit_sha=commit_response.json()["sha"],
            license_spdx=license_payload.get("spdx_id"),
            description=description,
            verified=verified,
            reference_files=reference_files,
            reference_excerpts=reference_excerpts,
        )
