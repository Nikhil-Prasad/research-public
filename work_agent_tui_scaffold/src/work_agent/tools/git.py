from __future__ import annotations

import subprocess
from pathlib import Path


def git_status(repo_root: Path) -> dict:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def git_diff(repo_root: Path, max_chars: int = 30000) -> dict:
    result = subprocess.run(
        ["git", "diff"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"returncode": result.returncode, "stdout_tail": result.stdout[-max_chars:], "stderr_tail": result.stderr[-max_chars:]}


def git_revert_worktree(repo_root: Path) -> dict:
    result = subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
