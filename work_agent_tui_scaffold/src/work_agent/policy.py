from __future__ import annotations

import fnmatch
import shlex
from pathlib import Path
from typing import Any

import yaml


class PolicyError(Exception):
    pass


class Policy:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def repo_root(self) -> Path:
        return Path(self.data["workspace"]["default_repo"]).resolve()

    def assert_under_allowed_root(self, path: Path) -> None:
        resolved = path.resolve()
        allowed = [Path(p).resolve() for p in self.data["workspace"].get("allowed_roots", [])]

        if not allowed:
            raise PolicyError("No allowed_roots configured")

        if not any(str(resolved).startswith(str(root)) for root in allowed):
            raise PolicyError(f"Path not under allowed roots: {resolved}")

    def _relative_to_repo(self, repo_root: Path, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError as exc:
            raise PolicyError(f"Path not under repo root: {path}") from exc

    def assert_file_read_allowed(self, repo_root: Path, path: Path) -> None:
        resolved = path.resolve()
        self.assert_under_allowed_root(resolved)
        rel = self._relative_to_repo(repo_root, resolved)

        for pattern in self.data["files"].get("blocked_globs", []):
            if fnmatch.fnmatch(rel, pattern):
                raise PolicyError(f"Blocked file path: {rel}")

        allowed = self.data["files"].get("readonly_globs", [])
        if not any(fnmatch.fnmatch(rel, pattern) for pattern in allowed):
            raise PolicyError(f"File type not readable by policy: {rel}")

    def assert_file_edit_allowed(self, repo_root: Path, path: Path) -> None:
        resolved = path.resolve()
        self.assert_under_allowed_root(resolved)
        rel = self._relative_to_repo(repo_root, resolved)

        for pattern in self.data["files"].get("blocked_globs", []):
            if fnmatch.fnmatch(rel, pattern):
                raise PolicyError(f"Blocked edit path: {rel}")

        allowed = self.data["files"].get("editable_globs", [])
        if not any(fnmatch.fnmatch(rel, pattern) for pattern in allowed):
            raise PolicyError(f"File not editable by policy: {rel}")

    def assert_shell_allowed(self, command: str) -> list[str]:
        if not self.data.get("shell", {}).get("enabled", True):
            raise PolicyError("Shell tool disabled")

        blocked = self.data["shell"].get("blocked_substrings", [])
        lowered = f" {command.lower()} "

        for token in blocked:
            if token.lower() in lowered:
                raise PolicyError(f"Blocked shell token: {token}")

        argv = shlex.split(command)
        if not argv:
            raise PolicyError("Empty shell command")

        executable = Path(argv[0]).name
        allowed = set(self.data["shell"].get("allowed_executables", []))

        if executable not in allowed:
            raise PolicyError(f"Executable not allowed: {executable}")

        if executable in {"python", "python3"} and len(argv) > 1:
            if argv[1] in {"-c", "-"}:
                raise PolicyError("Inline Python execution is blocked")

        if executable == "git" and len(argv) > 1:
            subcommand = argv[1]
            allowed_git = set(self.data["shell"].get("allowed_git_subcommands", []))
            if subcommand not in allowed_git:
                raise PolicyError(f"git subcommand not allowed via shell: {subcommand}")

        return argv

    def assert_sql_allowed(self, query: str) -> None:
        if not self.data.get("sql", {}).get("enabled", False):
            raise PolicyError("SQL tool disabled")

        lowered = query.lower()
        for token in self.data["sql"].get("blocked_keywords", []):
            if token.lower() in lowered:
                raise PolicyError(f"Blocked SQL keyword: {token}")

        stripped = lowered.strip()
        if not stripped.startswith(("select", "with", "explain")):
            raise PolicyError("Only SELECT/WITH/EXPLAIN SQL is allowed")

    def approval_required(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        cfg = self.data.get("approvals", {})
        if tool_name in set(cfg.get("require_tools", [])):
            return True

        if tool_name == "run_shell":
            command = str(tool_args.get("command", "")).lower()
            for token in cfg.get("require_shell_if_contains", []):
                if token.lower() in command:
                    return True

        return False
