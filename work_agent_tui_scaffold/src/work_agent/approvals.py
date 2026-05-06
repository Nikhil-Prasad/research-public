from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from work_agent.contracts import AgentEvent, ApprovalDecision, ApprovalRequest
from work_agent.policy import Policy


class ApprovalBroker:
    """Handles human approval for risky actions.

    Headless CLI mode can auto-approve or auto-reject based on policy.
    TUI mode passes an interactive_decider that blocks until the user presses approve/reject.
    """

    def __init__(
        self,
        policy: Policy,
        interactive_decider: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        self.policy = policy
        self.interactive_decider = interactive_decider
        self.on_event = on_event

    def maybe_approve(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        action_reason: str,
        step: int | None,
    ) -> ApprovalDecision | None:
        if not self.policy.approval_required(tool_name, tool_args):
            return None

        request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            tool_name=tool_name,
            tool_args=tool_args,
            reason=action_reason,
            risk="high" if tool_name in {"apply_patch", "run_sql"} else "medium",
        )

        if self.on_event:
            self.on_event(
                AgentEvent(
                    type="approval_required",
                    step=step,
                    message=f"Approval required for {tool_name}",
                    payload=request.model_dump(),
                )
            )

        if self.interactive_decider:
            decision = self.interactive_decider(request)
        else:
            default = self.policy.data.get("approvals", {}).get("headless_default", "approve")
            approved = default == "approve"
            decision = ApprovalDecision(
                request_id=request.request_id,
                approved=approved,
                reason=f"Headless default: {default}",
            )

        if self.on_event:
            self.on_event(
                AgentEvent(
                    type="approval_decision",
                    step=step,
                    message=("Approved" if decision.approved else "Rejected") + f" {tool_name}",
                    payload=decision.model_dump(),
                )
            )

        return decision
