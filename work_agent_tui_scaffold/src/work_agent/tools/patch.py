from __future__ import annotations

import re
import subprocess
from pathlib import Path

from work_agent.contracts import PatchResult
from work_agent.policy import Policy, PolicyError


DOC_IDISH = re.compile(r"\b(doc[_-]?\d{2,}|document[_-]?\d{2,}|file[_-]?\d{2,})\b", re.IGNORECASE)


def _changed_files_from_diff(diff: str) -> list[str]:
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            rel = line.removeprefix("+++ b/").strip()
            if rel != "/dev/null":
                files.append(rel)
    return files


def _changed_line_count(diff: str) -> int:
    return sum(1 for line in diff.splitlines() if line.startswith("+") or line.startswith("-"))


def apply_patch(*, unified_diff: str, repo_root: Path, policy: Policy) -> PatchResult:
    patch_policy = policy.data["patch"]

    if not patch_policy.get("enabled", True):
        return PatchResult(applied=False, reason="Patch tool disabled")

    if not unified_diff.startswith("diff --git "):
        return PatchResult(applied=False, reason="Expected git unified diff")

    changed_files = _changed_files_from_diff(unified_diff)

    if len(changed_files) > patch_policy.get("max_files_changed", 5):
        return PatchResult(applied=False, reason="Too many files changed")

    if _changed_line_count(unified_diff) > patch_policy.get("max_lines_changed", 500):
        return PatchResult(applied=False, reason="Too many lines changed")

    if patch_policy.get("forbid_doc_id_literals", True) and DOC_IDISH.search(unified_diff):
        return PatchResult(applied=False, reason="Patch appears to contain doc-id-like literals")

    try:
        for rel in changed_files:
            policy.assert_file_edit_allowed(repo_root, repo_root / rel)
    except PolicyError as exc:
        return PatchResult(applied=False, reason=str(exc))

    patch_path = repo_root / ".work_agent_candidate.patch"
    patch_path.write_text(unified_diff, encoding="utf-8")

    check = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    if check.returncode != 0:
        return PatchResult(applied=False, reason="git apply --check failed", stdout=check.stdout, stderr=check.stderr)

    applied = subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    return PatchResult(
        applied=applied.returncode == 0,
        reason="applied" if applied.returncode == 0 else "git apply failed",
        stdout=applied.stdout,
        stderr=applied.stderr,
    )
