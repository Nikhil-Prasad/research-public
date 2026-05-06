from __future__ import annotations

import json
import threading
from pathlib import Path
from threading import Event

from work_agent.approvals import ApprovalBroker
from work_agent.driver import AgentDriver
from work_agent.llm_client import BaseLLMClient
from work_agent.policy import Policy
from work_agent.run_store import RunStore
from work_agent.tui.state import TuiState


def _json_tail(obj: object, max_chars: int = 6000) -> str:
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    return text[-max_chars:]


class WorkAgentTui:
    def __init__(
        self,
        *,
        repo_root: Path,
        mode: str,
        policy: Policy,
        llm: BaseLLMClient,
        run_store: RunStore,
        task: str | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.task = task
        self.mode = mode
        self.policy = policy
        self.llm = llm
        self.run_store = run_store
        self.state = TuiState()
        self.stop_event = Event()
        self.agent_thread: threading.Thread | None = None
        self.app = None
        self.input_area = None
        self.events_window = None

    def run(self) -> str | None:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.filters import has_focus
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style
            from prompt_toolkit.widgets import Frame, TextArea
        except ImportError as exc:
            raise RuntimeError("Install TUI dependencies: uv sync") from exc

        self.input_area = TextArea(
            height=3,
            prompt="task> ",
            multiline=False,
            wrap_lines=True,
            accept_handler=self._on_submit,
        )

        self.events_window = Window(
            FormattedTextControl(self.render_events, focusable=True, show_cursor=False),
            wrap_lines=False,
        )

        kb = KeyBindings()
        input_focused = has_focus(self.input_area)

        @kb.add("c-c")
        @kb.add("c-q")
        def _quit(event):
            self.stop_event.set()
            event.app.exit(result=None)

        @kb.add("q", filter=~input_focused)
        def _q(event):
            self.stop_event.set()
            event.app.exit(result=None)

        @kb.add("s", filter=~input_focused)
        def _stop(event):
            self.stop_event.set()
            self.state.add_info("Stop requested by user")
            event.app.invalidate()

        @kb.add("a", filter=~input_focused)
        def _approve(event):
            self.state.resolve_approval(True, "Approved in TUI")
            event.app.invalidate()

        @kb.add("r", filter=~input_focused)
        def _reject(event):
            self.state.resolve_approval(False, "Rejected in TUI")
            event.app.invalidate()

        @kb.add("?", filter=~input_focused)
        def _help(event):
            self.state.toggle_help()
            event.app.invalidate()

        @kb.add("d", filter=~input_focused)
        def _details(event):
            self.state.toggle_detail_mode()
            event.app.invalidate()

        @kb.add("tab")
        def _toggle_focus(event):
            if input_focused():
                event.app.layout.focus(self.events_window)
            else:
                event.app.layout.focus(self.input_area)

        header = Window(
            FormattedTextControl(self.render_header),
            height=3,
            dont_extend_height=True,
        )
        events_frame = Frame(body=self.events_window, title="Events")
        detail = Frame(
            body=Window(FormattedTextControl(self.render_detail), wrap_lines=True),
            title="Latest / Details",
        )
        approval = Window(
            FormattedTextControl(self.render_approval),
            height=7,
            dont_extend_height=True,
        )
        input_frame = Frame(body=self.input_area, title="Type a task and press Enter (Tab toggles focus)")
        footer = Window(
            FormattedTextControl(self.render_footer),
            height=2,
            dont_extend_height=True,
        )

        root = HSplit([
            header,
            VSplit([events_frame, detail]),
            approval,
            input_frame,
            footer,
        ])

        style = Style.from_dict({
            "header": "reverse",
            "ok": "ansigreen",
            "bad": "ansired",
            "warn": "ansiyellow",
            "dim": "ansibrightblack",
            "title": "bold",
        })

        self.app = Application(
            layout=Layout(root, focused_element=self.input_area),
            key_bindings=kb,
            full_screen=True,
            refresh_interval=0.5,
            style=style,
        )
        self.state.set_invalidator(self.app.invalidate)

        if self.task:
            self._start_agent(self.task)

        return self.app.run()

    def _on_submit(self, buffer) -> bool:
        text = buffer.text.strip()
        if not text:
            return False
        if self.agent_thread and self.agent_thread.is_alive():
            self.state.add_info("Agent is already running — wait for it to finish.")
            return False
        self._start_agent(text)
        return False

    def _start_agent(self, task: str) -> None:
        self.task = task
        self.stop_event = Event()
        self.agent_thread = threading.Thread(target=self._run_agent, args=(task,), daemon=True)
        self.agent_thread.start()

    def _run_agent(self, task: str) -> None:
        approvals = ApprovalBroker(
            policy=self.policy,
            interactive_decider=self.state.ask_approval,
        )
        driver = AgentDriver(
            repo_root=self.repo_root,
            policy=self.policy,
            llm=self.llm,
            run_store=self.run_store,
            approvals=approvals,
            on_event=self.state.add_event,
            stop_event=self.stop_event,
        )
        driver.run(task=task, mode=self.mode)
        if self.app:
            self.app.invalidate()

    def render_header(self):
        snap = self.state.snapshot()
        status = snap["status"][:120]
        task_line = self.task or "(awaiting input — type below and press Enter)"
        return [
            ("class:header", f" work-agent TUI | mode={self.mode} | repo={self.repo_root}\n"),
            ("", f" Task: {task_line[:160]}\n"),
            ("class:dim", f" Status: {status}"),
        ]

    def render_events(self):
        snap = self.state.snapshot()
        lines = []
        for event in snap["events"][-30:]:
            marker = {
                "run_started": "▶",
                "model_started": "…",
                "model_action": "◇",
                "tool_started": "⚙",
                "tool_finished": "✓",
                "approval_required": "!",
                "approval_decision": "◆",
                "run_finished": "■",
                "error": "✗",
                "info": "i",
            }.get(event.type, "-")
            style = "class:bad" if event.type == "error" else "class:warn" if event.type.startswith("approval") else ""
            step = "-" if event.step is None else str(event.step)
            lines.append((style, f"{marker} [{step}] {event.type}: {event.message}\n"))
        if not lines:
            lines.append(("class:dim", "No events yet.\n"))
        return lines

    def render_detail(self):
        snap = self.state.snapshot()
        if snap["help_visible"]:
            return [("", HELP_TEXT)]

        pending = snap["pending_approval"]
        if pending:
            return [("class:warn", "Pending approval:\n"), ("", _json_tail(pending.request.model_dump(), 6000))]

        events = snap["events"]
        if not events:
            return [("class:dim", "Waiting for agent events...\n")]

        if snap["detail_mode"] == "events":
            payload = [e.model_dump() for e in events[-5:]]
        else:
            payload = events[-1].model_dump()
        return [("", _json_tail(payload, 8000))]

    def render_approval(self):
        snap = self.state.snapshot()
        pending = snap["pending_approval"]
        if not pending:
            return [("class:dim", "No approval pending. Tools run under policy.\n")]

        req = pending.request
        body = [
            ("class:warn", f"Approval required: {req.tool_name} | risk={req.risk}\n"),
            ("", f"Reason: {req.reason[:220]}\n"),
            ("", f"Args: {_json_tail(req.tool_args, 700)}\n"),
            ("class:title", "Tab to events pane, then 'a' approve / 'r' reject.\n"),
        ]
        return body

    def render_footer(self):
        return [
            ("class:header", " input: Enter submit | Tab toggle focus | Ctrl-Q quit  ||  events: q quit | s stop | a/r approve/reject | d detail | ? help\n"),
            ("class:dim", f" artifacts: {self.run_store.path}"),
        ]


HELP_TEXT = """
work-agent TUI

The TUI is only a view/controller around the headless AgentDriver.
The driver still owns the loop. The ToolRegistry still owns execution. policy.yaml still owns permissions.

Focus:
  Tab           toggle focus between input box and events pane

Input box (focused by default):
  Enter         submit task
  Ctrl-Q / Ctrl-C  quit

Events pane:
  q / Ctrl-C    quit UI and request stop
  s             request agent stop
  a             approve pending action
  r             reject pending action
  d             toggle latest event vs recent event JSON
  ?             toggle this help

Approval prompts appear for tools configured in policy.yaml, normally apply_patch and run_sql.
"""
