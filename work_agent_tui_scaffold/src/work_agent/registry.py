from __future__ import annotations

from pathlib import Path

from work_agent.approvals import ApprovalBroker
from work_agent.contracts import ToolResult
from work_agent.policy import Policy
from work_agent.tools.files import list_dir, read_file, search_files
from work_agent.tools.git import git_diff, git_status
from work_agent.tools.mcp import call_mcp
from work_agent.tools.patch import apply_patch
from work_agent.tools.shell import run_shell
from work_agent.tools.sql import run_sql


class ToolRegistry:
    def __init__(self, repo_root: Path, policy: Policy, approvals: ApprovalBroker | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.policy = policy
        self.approvals = approvals

    def call(self, tool_name: str, args: dict, *, action_reason: str = "", step: int | None = None) -> ToolResult:
        try:
            if self.approvals:
                decision = self.approvals.maybe_approve(
                    tool_name=tool_name,
                    tool_args=args,
                    action_reason=action_reason,
                    step=step,
                )
                if decision and not decision.approved:
                    return ToolResult(tool_name=tool_name, ok=False, error=f"Rejected by approval gate: {decision.reason}")

            if tool_name == "read_file":
                out = read_file(path=args["path"], repo_root=self.repo_root, policy=self.policy)

            elif tool_name == "list_dir":
                out = list_dir(path=args.get("path", "."), repo_root=self.repo_root, policy=self.policy)

            elif tool_name == "search_files":
                out = search_files(
                    query=args["query"],
                    glob=args.get("glob", "**/*"),
                    repo_root=self.repo_root,
                    policy=self.policy,
                    max_matches=int(args.get("max_matches", 100)),
                )

            elif tool_name == "run_shell":
                out = run_shell(
                    command=args["command"],
                    cwd=args.get("cwd", str(self.repo_root)),
                    timeout_seconds=args.get("timeout_seconds"),
                    repo_root=self.repo_root,
                    policy=self.policy,
                )

            elif tool_name == "git_status":
                out = git_status(self.repo_root)

            elif tool_name == "git_diff":
                out = git_diff(self.repo_root)

            elif tool_name == "apply_patch":
                out = apply_patch(unified_diff=args["unified_diff"], repo_root=self.repo_root, policy=self.policy)

            elif tool_name == "run_sql":
                out = run_sql(query=args["query"], policy=self.policy)

            elif tool_name == "call_mcp":
                out = call_mcp(
                    server=args["server"],
                    tool=args["tool"],
                    arguments=args.get("arguments", {}),
                    policy=self.policy,
                )

            else:
                return ToolResult(tool_name=tool_name, ok=False, error=f"Unknown tool: {tool_name}")

            result = out.model_dump() if hasattr(out, "model_dump") else out
            return ToolResult(tool_name=tool_name, ok=True, result=result)

        except Exception as exc:
            return ToolResult(tool_name=tool_name, ok=False, error=str(exc))
