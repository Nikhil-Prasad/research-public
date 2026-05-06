from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentAction(StrictModel):
    """
    One model step.

    The model must either:
    - call exactly one tool, or
    - finish with a final answer/report.
    """

    action: Literal["tool_call", "finish"]
    reason: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None


class ToolResult(StrictModel):
    tool_name: str
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AgentObservation(StrictModel):
    step: int
    action: AgentAction
    result: ToolResult


class AgentState(StrictModel):
    task: str
    repo_root: str
    mode: str = "patch-assist"
    observations: list[AgentObservation] = Field(default_factory=list)


class AgentEvent(StrictModel):
    type: Literal[
        "run_started",
        "model_started",
        "model_action",
        "tool_started",
        "tool_finished",
        "approval_required",
        "approval_decision",
        "run_finished",
        "error",
        "info",
    ]
    step: int | None = None
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(StrictModel):
    request_id: str
    tool_name: str
    tool_args: dict[str, Any]
    reason: str
    risk: Literal["low", "medium", "high"] = "medium"


class ApprovalDecision(StrictModel):
    request_id: str
    approved: bool
    reason: str


class ShellResult(StrictModel):
    command: str
    cwd: str
    returncode: int
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False


class FileReadResult(StrictModel):
    path: str
    content: str
    truncated: bool = False


class SearchResult(StrictModel):
    query: str
    matches: list[dict[str, Any]]


class PatchResult(StrictModel):
    applied: bool
    reason: str
    stdout: str = ""
    stderr: str = ""


class SqlResult(StrictModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class McpResult(StrictModel):
    server: str
    tool: str
    output: dict[str, Any]
