from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable

from work_agent.contracts import AgentEvent, ApprovalDecision, ApprovalRequest


@dataclass
class PendingApproval:
    request: ApprovalRequest
    event: Event
    decision: ApprovalDecision | None = None


class TuiState:
    def __init__(self) -> None:
        self._lock = Lock()
        self.events: list[AgentEvent] = []
        self.status: str = "idle"
        self.final_report: str = ""
        self.pending_approval: PendingApproval | None = None
        self.help_visible: bool = False
        self.detail_mode: str = "latest"
        self.invalidate: Callable[[], None] | None = None

    def set_invalidator(self, invalidate: Callable[[], None]) -> None:
        self.invalidate = invalidate

    def _invalidate(self) -> None:
        if self.invalidate:
            self.invalidate()

    def add_event(self, event: AgentEvent) -> None:
        with self._lock:
            self.events.append(event)
            if len(self.events) > 500:
                self.events = self.events[-500:]
            self.status = event.message
            if event.type == "run_finished":
                self.final_report = str(event.payload.get("final_answer", event.message))
        self._invalidate()

    def ask_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        wait_event = Event()
        pending = PendingApproval(request=request, event=wait_event)
        with self._lock:
            self.pending_approval = pending
            self.status = f"Approval pending: {request.tool_name}"
        self._invalidate()
        wait_event.wait()
        with self._lock:
            decision = pending.decision
            self.pending_approval = None
        self._invalidate()
        if decision is None:
            return ApprovalDecision(request_id=request.request_id, approved=False, reason="No decision recorded")
        return decision

    def resolve_approval(self, approved: bool, reason: str) -> None:
        with self._lock:
            pending = self.pending_approval
            if not pending:
                return
            pending.decision = ApprovalDecision(
                request_id=pending.request.request_id,
                approved=approved,
                reason=reason,
            )
            pending.event.set()
        self._invalidate()

    def add_info(self, message: str) -> None:
        self.add_event(AgentEvent(type="info", message=message, payload={}))

    def toggle_help(self) -> None:
        with self._lock:
            self.help_visible = not self.help_visible
        self._invalidate()

    def toggle_detail_mode(self) -> None:
        with self._lock:
            self.detail_mode = "events" if self.detail_mode == "latest" else "latest"
        self._invalidate()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "events": list(self.events),
                "status": self.status,
                "final_report": self.final_report,
                "pending_approval": self.pending_approval,
                "help_visible": self.help_visible,
                "detail_mode": self.detail_mode,
            }
