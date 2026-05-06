from __future__ import annotations

from pathlib import Path

from work_agent.contracts import FileReadResult, SearchResult
from work_agent.policy import Policy


def _resolve_repo_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def read_file(*, path: str, repo_root: Path, policy: Policy) -> FileReadResult:
    target = _resolve_repo_path(repo_root, path)
    policy.assert_file_read_allowed(repo_root, target)

    max_chars = policy.data["files"].get("max_read_chars", 20000)
    content = target.read_text(encoding="utf-8", errors="replace")

    return FileReadResult(
        path=str(target.relative_to(repo_root)),
        content=content[:max_chars],
        truncated=len(content) > max_chars,
    )


def list_dir(*, path: str, repo_root: Path, policy: Policy) -> dict:
    target = _resolve_repo_path(repo_root, path)
    policy.assert_under_allowed_root(target)

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            rel = str(child.relative_to(repo_root))
        except ValueError:
            rel = str(child)
        entries.append({"name": child.name, "path": rel, "is_dir": child.is_dir()})

    try:
        path_out = str(target.relative_to(repo_root))
    except ValueError:
        path_out = str(target)

    return {"path": path_out, "entries": entries}


def search_files(
    *,
    query: str,
    repo_root: Path,
    policy: Policy,
    glob: str = "**/*",
    max_matches: int = 100,
) -> SearchResult:
    matches = []
    q = query.lower()

    for path in repo_root.glob(glob):
        if not path.is_file():
            continue

        try:
            policy.assert_file_read_allowed(repo_root, path)
        except Exception:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if q in line.lower():
                matches.append({"path": str(path.relative_to(repo_root)), "line": line_no, "text": line[:500]})
                if len(matches) >= max_matches:
                    return SearchResult(query=query, matches=matches)

    return SearchResult(query=query, matches=matches)
