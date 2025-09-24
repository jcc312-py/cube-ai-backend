# repo_importer.py
from __future__ import annotations
import os
import re
import io
import zipfile
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional, List

import requests
from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl, ConfigDict

router = APIRouter(tags=["repository-import"])

# Root workspace where repos will be stored
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Allow-list of hosts (security: do not clone arbitrary hosts)
ALLOWED_HOSTS = {"github.com"}

# --- Models ------------------------------------------------------------------

class RepoImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_url: HttpUrl = Field(..., description="HTTPS URL to the GitHub repository")
    branch: Optional[str] = Field(None, description="Branch to checkout (default: repo default)")
    method: Literal["git", "zip"] = Field(
        "git", description="Clone via 'git' or download GitHub ZIP via API"
    )
    subdir_name: Optional[str] = Field(
        None,
        description="Optional directory name to place repo into. Defaults to <owner>__<repo>__<ts>",
    )

class RepoInfo(BaseModel):
    id: str
    name: str
    path: str
    branch: Optional[str] = None
    imported_at: float
    method: str

class RepoListResponse(BaseModel):
    repos: List[RepoInfo]

class RepoDeleteResponse(BaseModel):
    deleted: bool
    id: str

# --- Utility -----------------------------------------------------------------

def _now() -> float:
    return time.time()

def _validate_github_url(url: str) -> tuple[str, str]:
    """
    Validate the URL host and parse owner/repo from typical forms:
      https://github.com/<owner>/<repo>
      https://github.com/<owner>/<repo>.git
    Returns (owner, repo)
    """
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid repo_url")

    host = u.netloc.lower()
    if host not in ALLOWED_HOSTS:
        raise HTTPException(status_code=400, detail=f"Host '{host}' is not allowed")

    # path like "/owner/repo" or "/owner/repo.git"
    m = re.match(r"^/([^/]+)/([^/]+?)(?:\.git)?/?$", u.path)
    if not m:
        raise HTTPException(status_code=400, detail="repo_url must be like https://github.com/<owner>/<repo>")
    owner, repo = m.group(1), m.group(2)
    return owner, repo

def _sanitize_dir_name(name: str) -> str:
    # keep alnum, dash, underscore; map others to underscore
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

def _new_repo_dir(owner: str, repo: str, subdir_name: Optional[str]=None) -> Path:
    ts = int(_now())
    base = subdir_name or f"{owner}__{repo}__{ts}"
    safe = _sanitize_dir_name(base)
    target = WORKSPACE_DIR / safe
    # Avoid collision just in case
    i = 0
    while target.exists():
        i += 1
        target = WORKSPACE_DIR / f"{safe}_{i}"
    return target

def _mask_token(token: Optional[str]) -> str:
    if not token:
        return ""
    if len(token) <= 6:
        return "*" * len(token)
    return token[:3] + "*" * (len(token)-6) + token[-3:]

def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

# --- Core import implementations --------------------------------------------

def _clone_with_git(
    repo_url: str,
    branch: Optional[str],
    dest: Path,
    token: Optional[str],
) -> RepoInfo:
    """
    Secure-ish git clone via subprocess without shell=True.
    We optionally pass a header for private access via token.
    """
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    args = ["git", "clone", "--depth", "1"]
    if branch:
        args += ["--branch", branch]

    # For private repos over HTTPS, add Authorization header
    # (Avoid putting token directly in URL; use http.extraHeader)
    if token:
        env["GIT_HTTP_EXTRA_HEADER"] = f"Authorization: Bearer {token}"

    args += [repo_url, str(dest)]

    try:
        r = subprocess.run(args, env=env, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400,
            detail=f"git clone failed: {e.stderr.strip() or e.stdout.strip()}"
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="git clone timed out")

    # Optionally clear the header from env for safety
    env.pop("GIT_HTTP_EXTRA_HEADER", None)

    info = RepoInfo(
        id=dest.name,
        name=dest.name,
        path=str(dest),
        branch=branch,
        imported_at=_now(),
        method="git",
    )
    return info

def _download_zip_via_api(
    owner: str,
    repo: str,
    branch: Optional[str],
    dest: Path,
    token: Optional[str],
) -> RepoInfo:
    """
    Download a ZIP archive from GitHub and extract it. Works for public repos;
    for private, requires a token with repo read perms.
    """
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    ref = branch or "HEAD"
    # The GitHub API for archive zips:
    #   GET https://api.github.com/repos/{owner}/{repo}/zipball/{ref}
    api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{ref}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with requests.get(api_url, headers=headers, stream=True, timeout=60) as resp:
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text[:200]}")
            content = io.BytesIO(resp.content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download repo zip: {e}")

    # Extract zip into temp then move to dest (zip contains a top-level folder)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(content) as zf:
                zf.extractall(tmpdir_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract repo zip: {e}")

        # Find the top-level directory extracted by GitHub
        entries = [p for p in tmpdir_path.iterdir() if p.is_dir()]
        if not entries:
            raise HTTPException(status_code=500, detail="ZIP contained no directories")
        top = entries[0]

        # Move into final destination
        shutil.move(str(top), str(dest))

    info = RepoInfo(
        id=dest.name,
        name=dest.name,
        path=str(dest),
        branch=branch,
        imported_at=_now(),
        method="zip",
    )
    return info

# --- Repo Registry (simple, file-less) --------------------------------------

def _load_repos() -> List[RepoInfo]:
    repos: List[RepoInfo] = []
    if not WORKSPACE_DIR.exists():
        return repos
    for p in WORKSPACE_DIR.iterdir():
        if p.is_dir():
            # Minimal info; more metadata can be stored in a sidecar JSON if needed
            repos.append(
                RepoInfo(
                    id=p.name,
                    name=p.name,
                    path=str(p.resolve()),
                    branch=None,
                    imported_at=p.stat().st_mtime,
                    method="unknown",
                )
            )
    return sorted(repos, key=lambda r: r.imported_at, reverse=True)

def _delete_repo(repo_id: str) -> bool:
    target = (WORKSPACE_DIR / repo_id).resolve()
    if WORKSPACE_DIR not in target.parents:
        # Safety: prevent deleting outside workspace
        raise HTTPException(status_code=400, detail="Invalid repo id")
    if target.exists():
        shutil.rmtree(target)
        return True
    return False

# --- Background task wrapper (nice for big repos) ----------------------------

def _background_cleanup(path: Path, keep_if_exists: bool):
    # Placeholder for any async cleanup or indexing hooks
    # For example: precompute file tree, index code, etc.
    if keep_if_exists and not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# --- Routes ------------------------------------------------------------------

@router.post("/repos/import", response_model=RepoInfo, summary="Import a GitHub repository")
def import_repo(
    payload: RepoImportRequest,
    background: BackgroundTasks,
    x_github_token: Optional[str] = Header(None, convert_underscores=False, description="Optional token for private repos"),
):
    owner, repo = _validate_github_url(str(payload.repo_url))
    dest = _new_repo_dir(owner, repo, payload.subdir_name)
    token_msg = _mask_token(x_github_token)

    if payload.method == "git":
        if not _git_available():
            raise HTTPException(status_code=500, detail="git is not available in the runtime")
        info = _clone_with_git(str(payload.repo_url), payload.branch, dest, x_github_token)
    else:
        info = _download_zip_via_api(owner, repo, payload.branch, dest, x_github_token)

    # Run any optional background cleanup/indexing without blocking the request
    background.add_task(_background_cleanup, dest, False)

    # (No logging of token; masked only if needed)
    return info

@router.get("/repos", response_model=RepoListResponse, summary="List imported repositories")
def list_repos():
    return RepoListResponse(repos=_load_repos())

@router.delete("/repos/{repo_id}", response_model=RepoDeleteResponse, summary="Delete an imported repository")
def delete_repo(repo_id: str):
    deleted = _delete_repo(repo_id)
    return RepoDeleteResponse(deleted=deleted, id=repo_id)
