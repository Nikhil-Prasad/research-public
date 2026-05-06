from __future__ import annotations

import subprocess
from pathlib import Path

from work_agent.contracts import ShellResult
from work_agent.policy import Policy


def resolve_cwd(cwd: str | None, repo_root: Path) -> Path:
    if not cwd:
        return repo_root.resolve()
    candidate = Path(cwd)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def run_shell(
    *,
    command: str,
    cwd: str | None,
    policy: Policy,
    repo_root: Path,
    timeout_seconds: int | None = None,
) -> ShellResult:
    argv = policy.assert_shell_allowed(command)
    cwd_path = resolve_cwd(cwd, repo_root)
    policy.assert_under_allowed_root(cwd_path)

    timeout = timeout_seconds or policy.data["shell"].get("timeout_seconds", 600)
    max_chars = policy.data["agent"].get("max_observation_chars", 12000)

    try:
        result = subprocess.run(
            argv,
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        return ShellResult(
            command=command,
            cwd=str(cwd_path),
            returncode=result.returncode,
            stdout_tail=result.stdout[-max_chars:],
            stderr_tail=result.stderr[-max_chars:],
            timed_out=False,
        )

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ShellResult(
            command=command,
            cwd=str(cwd_path),
            returncode=-1,
            stdout_tail=stdout[-max_chars:],
            stderr_tail=stderr[-max_chars:],
            timed_out=True,
        )
