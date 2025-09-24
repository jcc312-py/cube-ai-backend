from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Literal, Optional, List

import requests
from fastapi import APIRouter, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl, ConfigDict

# -----------------------------------------------------------------------------
# Router & Workspace
# -----------------------------------------------------------------------------
router = APIRouter(tags=["repository-import"])

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_HOSTS = {"github.com"}  # safety: only allow GitHub

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RepoImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_url: HttpUrl = Field(..., description="HTTPS URL to the GitHub repository")
    branch: Optional[str] = Field(None, description="Branch to checkout (default: repo default)")
    method: Literal["git", "zip"] = Field("git", description="Clone via 'git' or fetch ZIP via GitHub API")
    subdir_name: Optional[str] = Field(
        None,
        description="Optional directory name for the imported repo; defaults to <owner>__<repo>__<ts>",
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

# Push / Status
class PushRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: str = Field("ai-updates", description="Target branch to push to (created/reset to current state)")
    commit_message: str = Field("Update via API", description="Commit message")
    add_all: bool = Field(True, description="Stage all changes before committing")
    force: bool = Field(False, description="Force push (use sparingly)")

class PushResponse(BaseModel):
    id: str
    branch: str
    committed: bool
    pushed: bool
    stdout: str
    stderr: str

class RepoStatus(BaseModel):
    id: str
    branch: str
    dirty: bool
    ahead: Optional[int] = None
    behind: Optional[int] = None

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _now() -> float:
    return time.time()

def _validate_github_url(url: str) -> tuple[str, str]:
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid repo_url")

    host = u.netloc.lower()
    if host not in ALLOWED_HOSTS:
        raise HTTPException(status_code=400, detail=f"Host '{host}' is not allowed")

    # /owner/repo or /owner/repo.git
    m = re.match(r"^/([^/]+)/([^/]+?)(?:\.git)?/?$", u.path)
    if not m:
        raise HTTPException(status_code=400, detail="repo_url must look like https://github.com/<owner>/<repo>")
    owner, repo = m.group(1), m.group(2)
    return owner, repo

def _sanitize_dir_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

def _new_repo_dir(owner: str, repo: str, subdir_name: Optional[str]=None) -> Path:
    ts = int(_now())
    base = subdir_name or f"{owner}__{repo}__{ts}"
    safe = _sanitize_dir_name(base)
    target = WORKSPACE_DIR / safe
    i = 0
    while target.exists():
        i += 1
        target = WORKSPACE_DIR / f"{safe}_{i}"
    return target

def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def _run_git(repo_path: Path, args: list[str], env: Optional[dict]=None, timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=True, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or str(e)
        raise HTTPException(status_code=400, detail=f"git {' '.join(args)} failed: {msg}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail=f"git {' '.join(args)} timed out")

def _ensure_repo_exists(repo_id: str) -> Path:
    repo_path = (WORKSPACE_DIR / repo_id).resolve()
    if not repo_path.exists() or not (repo_path / ".git").exists():
        raise HTTPException(status_code=404, detail="Repository not found or not a git repo")
    # prevent path traversal
    if WORKSPACE_DIR not in repo_path.parents and repo_path != WORKSPACE_DIR:
        raise HTTPException(status_code=400, detail="Invalid repo id path")
    return repo_path

def _has_changes(repo_path: Path) -> bool:
    cp = _run_git(repo_path, ["status", "--porcelain"])
    return bool(cp.stdout.strip())

def _current_branch(repo_path: Path) -> str:
    cp = _run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    return cp.stdout.strip()

# -----------------------------------------------------------------------------
# Import Implementations
# -----------------------------------------------------------------------------
def _clone_with_git(repo_url: str, branch: Optional[str], dest: Path, token: Optional[str]) -> RepoInfo:
    dest.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    args = ["git", "clone", "--depth", "1"]
    if branch:
        args += ["--branch", branch]

    # private HTTPS: pass token via header (safer than embedding in URL)
    if token:
        env["GIT_HTTP_EXTRA_HEADER"] = f"Authorization: Bearer {token}"

    args += [repo_url, str(dest)]
    try:
        subprocess.run(args, env=env, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"git clone failed: {e.stderr.strip() or e.stdout.strip()}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="git clone timed out")
    finally:
        if "GIT_HTTP_EXTRA_HEADER" in env:
            env.pop("GIT_HTTP_EXTRA_HEADER", None)

    return RepoInfo(
        id=dest.name,
        name=dest.name,
        path=str(dest),
        branch=branch,
        imported_at=_now(),
        method="git",
    )

def _download_zip_via_api(owner: str, repo: str, branch: Optional[str], dest: Path, token: Optional[str]) -> RepoInfo:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = branch or "HEAD"
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(content) as zf:
                zf.extractall(tmpdir_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract repo zip: {e}")

        entries = [p for p in tmpdir_path.iterdir() if p.is_dir()]
        if not entries:
            raise HTTPException(status_code=500, detail="ZIP contained no directories")
        top = entries[0]
        shutil.move(str(top), str(dest))

    return RepoInfo(
        id=dest.name,
        name=dest.name,
        path=str(dest),
        branch=branch,
        imported_at=_now(),
        method="zip",
    )

# -----------------------------------------------------------------------------
# Registry (list / delete)
# -----------------------------------------------------------------------------
def _load_repos() -> List[RepoInfo]:
    if not WORKSPACE_DIR.exists():
        return []
    repos: List[RepoInfo] = []
    for p in WORKSPACE_DIR.iterdir():
        if p.is_dir():
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
        raise HTTPException(status_code=400, detail="Invalid repo id")
    if target.exists():
        shutil.rmtree(target)
        return True
    return False

# -----------------------------------------------------------------------------
# Routes: Import / List / Delete
# -----------------------------------------------------------------------------
@router.post("/repos/import", response_model=RepoInfo, summary="Import a GitHub repository")
def import_repo(
    payload: RepoImportRequest,
    background: BackgroundTasks,
    x_github_token: Optional[str] = Header(None, convert_underscores=False, description="Optional token for private repos"),
):
    owner, repo = _validate_github_url(str(payload.repo_url))
    dest = _new_repo_dir(owner, repo, payload.subdir_name)

    if payload.method == "git":
        if not _git_available():
            raise HTTPException(status_code=500, detail="git is not available in the runtime")
        info = _clone_with_git(str(payload.repo_url), payload.branch, dest, x_github_token)
    else:
        info = _download_zip_via_api(owner, repo, payload.branch, dest, x_github_token)

    # background hook (placeholder for indexing, etc.)
    background.add_task(lambda: None)
    return info

@router.get("/repos", response_model=RepoListResponse, summary="List imported repositories")
def list_repos():
    return RepoListResponse(repos=_load_repos())

@router.delete("/repos/{repo_id}", response_model=RepoDeleteResponse, summary="Delete an imported repository")
def delete_repo(repo_id: str):
    deleted = _delete_repo(repo_id)
    return RepoDeleteResponse(deleted=deleted, id=repo_id)

# -----------------------------------------------------------------------------
# Routes: Push & Status
# -----------------------------------------------------------------------------
def _checkout_branch(repo_path: Path, branch: str) -> None:
    # reset/initialize branch to current worktree (like checkout -B)
    _run_git(repo_path, ["checkout", "-B", branch])

@router.post("/repos/{repo_id}/push", response_model=PushResponse, summary="Commit and push changes to GitHub")
def push_repo_changes(
    repo_id: str,
    payload: PushRequest,
    x_github_token: Optional[str] = Header(None, convert_underscores=False, description="Optional token for pushing over HTTPS remotes"),
):
    """
    Commits local changes and pushes to origin/<branch>.
    Auth:
      - SSH remote: works if host has an SSH key with push access.
      - HTTPS remote: pass a PAT via X-GitHub-Token; we set GIT_HTTP_EXTRA_HEADER.
    Safe by default: pushes to a feature branch (default 'ai-updates').
    """
    repo_path = _ensure_repo_exists(repo_id)

    # local commit identity (doesn't affect global git config)
    _run_git(repo_path, ["config", "user.name", "CubeAI Bot"])
    _run_git(repo_path, ["config", "user.email", "bot@example.com"])

    committed = False
    if payload.add_all:
        _run_git(repo_path, ["add", "--all"])

    if _has_changes(repo_path):
        _run_git(repo_path, ["commit", "-m", payload.commit_message])
        committed = True

    _checkout_branch(repo_path, payload.branch)

    env = os.environ.copy()
    if x_github_token:
        env["GIT_HTTP_EXTRA_HEADER"] = f"Authorization: Bearer {x_github_token}"

    push_args = ["push", "-u", "origin", payload.branch]
    if payload.force:
        push_args.insert(1, "--force")

    cp = _run_git(repo_path, push_args, env=env, timeout=180)

    env.pop("GIT_HTTP_EXTRA_HEADER", None)
    return PushResponse(
        id=repo_id,
        branch=payload.branch,
        committed=committed,
        pushed=True,
        stdout=cp.stdout.strip(),
        stderr=cp.stderr.strip(),
    )

@router.get("/repos/{repo_id}/status", response_model=RepoStatus, summary="Show repo branch & change status")
def repo_status(repo_id: str):
    repo_path = _ensure_repo_exists(repo_id)
    branch = _current_branch(repo_path)
    dirty = _has_changes(repo_path)

    ahead_count = behind_count = None
    try:
        _run_git(repo_path, ["fetch", "origin", branch], timeout=30)
        counts = _run_git(repo_path, ["rev-list", "--left-right", "--count", f"origin/{branch}...{branch}"]).stdout.strip()
        left, right = [int(x) for x in counts.split()]
        # "left" = behind, "right" = ahead (relative to origin/branch)
        behind_count, ahead_count = left, right
    except Exception:
        pass

    return RepoStatus(id=repo_id, branch=branch, dirty=dirty, ahead=ahead_count, behind=behind_count)
