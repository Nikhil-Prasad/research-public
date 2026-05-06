from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from threading import Event

from work_agent.approvals import ApprovalBroker
from work_agent.contracts import AgentEvent, AgentObservation, AgentState
from work_agent.llm_client import BaseLLMClient
from work_agent.policy import Policy
from work_agent.registry import ToolRegistry
from work_agent.run_store import RunStore


class AgentDriver:
    def __init__(
        self,
        *,
        repo_root: Path,
        policy: Policy,
        llm: BaseLLMClient,
        run_store: RunStore,
        approvals: ApprovalBroker | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        stop_event: Event | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.policy = policy
        self.llm = llm
        self.run_store = run_store
        self.on_event = on_event
        self.stop_event = stop_event or Event()
        self.approvals = approvals or ApprovalBroker(policy)
        self.approvals.on_event = self.emit
        self.tools = ToolRegistry(self.repo_root, policy, approvals=self.approvals)

    def emit(self, event: AgentEvent) -> None:
        self.run_store.write_event(event.type, event.model_dump())
        if self.on_event:
            self.on_event(event)

    def run(self, task: str, mode: str = "patch-assist") -> str:
        state = AgentState(task=task, repo_root=str(self.repo_root), mode=mode)
        max_steps = int(self.policy.data["agent"].get("max_steps", 30))

        self.emit(AgentEvent(type="run_started", message="Agent run started", payload={"repo_root": str(self.repo_root), "mode": mode, "task": task}))

        for step in range(1, max_steps + 1):
            if self.stop_event.is_set():
                final = "Stopped by user."
                self.emit(AgentEvent(type="run_finished", step=step, message=final, payload={"stopped": True}))
                self.run_store.write_text("final_report.md", final)
                return final

            try:
                state_summary = self._summarize_state(state)
                self.emit(AgentEvent(type="model_started", step=step, message="Requesting next model action", payload={}))
                action = self.llm.next_action(task=task, state_summary=state_summary)
                self.run_store.write_json(f"step_{step:03d}_action.json", action.model_dump())
                self.emit(AgentEvent(type="model_action", step=step, message=action.reason, payload=action.model_dump()))

                if action.action == "finish":
                    final = action.final_answer or "Finished."
                    self.run_store.write_text("final_report.md", final)
                    self.emit(AgentEvent(type="run_finished", step=step, message="Agent finished", payload={"final_answer": final}))
                    return final

                if not action.tool_name:
                    result = self.tools.call("__invalid__", {"error": "tool_name missing"}, action_reason=action.reason, step=step)
                else:
                    self.emit(AgentEvent(type="tool_started", step=step, message=f"Running tool: {action.tool_name}", payload=action.tool_args))
                    result = self.tools.call(action.tool_name, action.tool_args, action_reason=action.reason, step=step)
                    self.emit(AgentEvent(type="tool_finished", step=step, message=f"Finished tool: {action.tool_name}", payload=result.model_dump()))

                observation = AgentObservation(step=step, action=action, result=result)
                state.observations.append(observation)
                self.run_store.write_json(f"step_{step:03d}_observation.json", observation.model_dump())

            except Exception as exc:
                message = f"Agent error at step {step}: {exc}"
                self.emit(AgentEvent(type="error", step=step, message=message, payload={"error": str(exc)}))
                self.run_store.write_text("final_report.md", message)
                return message

        final = "Stopped: max steps reached."
        self.run_store.write_text("final_report.md", final)
        self.emit(AgentEvent(type="run_finished", step=max_steps, message=final, payload={"max_steps": max_steps}))
        return final

    def _summarize_state(self, state: AgentState) -> str:
        max_chars = int(self.policy.data["agent"].get("max_observation_chars", 12000))
        payload = {
            "repo_root": state.repo_root,
            "mode": state.mode,
            "recent_observations": [obs.model_dump() for obs in state.observations[-8:]],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return text[-max_chars:]
