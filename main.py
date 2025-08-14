# assertion_verifier.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- PTA base (adjust import to your repo) ---
# Expecting something like:
# class BasePTAAgent(ABC):
#   async def perceive(self, user_input: str) -> Any: ...
#   async def think(self, perceived: Any) -> Any: ...
#   async def act(self, thought: Any) -> str: ...
#   async def run(self, user_input: str) -> str: ...
from llm_agents.core.base_agent import BasePTAAgent  # <- ensure your PTA ABC path

# --- Burr (from /burr/... per the supplied tree) ---
from burr.core import ApplicationBuilder, State, default
from burr.core.action import action

# --- MCP ToolBus only for catalog fetch if you want (no execution in Stage-1) ---
# from llm_agents.integrations.mcp_toolbus import ToolBus

from .assertion_verifier_models import (
    AssertionNode,
    PlanForAssertion,
    ToolRef,
    VerificationPlanReport,
    Complexity,
    VerificationStatus,
    VerificationStep,
    Priority,
    AssertionKind,
)

logger = logging.getLogger(__name__)


# ------------------------- small utils -------------------------

def _first_json_obj(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from text (handles codefences)."""
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text, flags=re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))
    obj = re.search(r"\{(?:[^{}]|(?R))*\}", text)  # not fully portable; last resort
    if obj:
        return json.loads(obj.group(0))
    raise ValueError("No JSON object found in LLM output.")


def _limit(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


@dataclass
class PlannerSettings:
    max_assertion_nodes: int = 32
    max_steps_per_assertion: int = 3
    route_simple: str = "plan"  # 'plan' | 'skip'
    prefer_authoritative: bool = True


# ------------------------- Burr actions (module-scope; bind resources) -------------------------

@action(reads=["input_text"], writes=["assertion_tree"])
async def perceive_extract_assertions(
    state: State, *, llm: Any, assertion_schema: Dict[str, Any]
) -> Tuple[Dict[str, Any], State]:
    """
    LLM → JSON: build a single-root AssertionNode (A1) with nested children.
    """
    doc_text: str = state["input_text"]
    prompt = (
        "You are a verification planner.\n"
        "Task: Extract verifiable assertions from the document as ONE JSON object matching this schema:\n"
        f"{json.dumps(assertion_schema, indent=2)}\n\n"
        "Rules:\n"
        "- Return exactly one root node (id 'A1') with nested children (A1.1, A1.2, ...).\n"
        "- Mark 'complexity' = 'complex' only if the node includes multiple sub-claims.\n"
        f"- Set 'kind' conservatively: one of {', '.join(k.value for k in AssertionKind)}.\n"
        "- Use 'priority' high/medium/low based on impact/risk.\n"
        "- Do NOT include any text outside the JSON.\n\n"
        f"Document (truncated):\n'''{_limit(doc_text, 8000)}'''\n"
    )

    try:
        # Prefer a structured generation if available
        if hasattr(llm, "generate_json"):
            obj = await llm.generate_json(prompt, schema=assertion_schema)  # type: ignore
        else:
            text = await llm.generate(prompt)
            obj = _first_json_obj(text)
        node = AssertionNode.model_validate(obj)
    except Exception as e:
        logger.warning("Assertion extraction failed, falling back. err=%s", e)
        # Fallback: naïve sentence split to children
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc_text) if s.strip()]
        children: List[AssertionNode] = []
        for i, s in enumerate(sentences[: 10], start=1):
            children.append(
                AssertionNode(
                    id=f"A1.{i}",
                    statement=s,
                    complexity=Complexity.simple,
                    status=VerificationStatus.planned,
                )
            )
        node = AssertionNode(
            id="A1",
            statement="Document assertions",
            kind=AssertionKind.generic,
            complexity=Complexity.complex if len(children) > 1 else Complexity.simple,
            status=VerificationStatus.planned,
            children=children,
        )

    return {"assertion_tree": node.model_dump()}, state.update(assertion_tree=node.model_dump())


@action(reads=["assertion_tree", "input_text", "tool_catalog"], writes=["plans"])
async def plan_steps_for_assertions(
    state: State, *, llm: Any, plan_schema: Dict[str, Any], max_steps_per_assertion: int
) -> Tuple[Dict[str, Any], State]:
    """
    For each AssertionNode, propose 1-3 step plans mapped to known tools.
    """
    root = AssertionNode.model_validate(state["assertion_tree"])
    catalog: List[Dict[str, Any]] = state.get("tool_catalog", [])
    catalog_str = json.dumps(catalog, indent=2)
    nodes = _flatten_nodes(root)

    all_plans: List[PlanForAssertion] = []
    for node in nodes:
        node_str = json.dumps(node.model_dump(), indent=2)
        rules = [
            "Prefer authoritative internal sources.",
            f"Produce 1-{max_steps_per_assertion} steps.",
            "Minimal steps; one strong step > many weak ones.",
            "Use 'inputs_template' keys that match the tool's input schema when known.",
            "Set 'expected_signal' and 'negative_signal' concisely.",
            "Set 'step_id' = '<assertion_id>-S<index>' (1-based).",
        ]
        prompt = (
            "You are a verification planner.\n"
            "Task: Plan concrete tool-backed steps for ONE assertion node as JSON matching this schema:\n"
            f"{json.dumps(plan_schema, indent=2)}\n\n"
            f"TOOL_CATALOG:\n{catalog_str}\n\n"
            "Rules:\n- " + "\n- ".join(rules) + "\n\n"
            f"ASSERTION_NODE:\n{node_str}\n"
        )
        try:
            if hasattr(llm, "generate_json"):
                obj = await llm.generate_json(prompt, schema=plan_schema)  # type: ignore
            else:
                text = await llm.generate(prompt)
                obj = _first_json_obj(text)
            plan = PlanForAssertion.model_validate(obj)
            # Normalize ids when missing
            for i, s in enumerate(plan.steps, start=1):
                if not s.step_id:
                    s.step_id = f"{node.id}-S{i}"
                if not s.assertion_id:
                    s.assertion_id = node.id
            all_plans.append(plan)
        except Exception as e:
            logger.warning("Planning failed for %s; using heuristic. err=%s", node.id, e)
            guess_tool = _guess_tool(node.statement, catalog)
            step = VerificationStep(
                step_id=f"{node.id}-S1",
                assertion_id=node.id,
                tool_name=guess_tool or (catalog[0]["name"] if catalog else "unknown"),
                rationale="Heuristic fallback due to planner error.",
                inputs_template={"query": node.statement},
                expected_signal="Records/documents that support the claim.",
                negative_signal="Records/documents that contradict the claim.",
                priority=node.priority,
                confidence=0.3,
            )
            all_plans.append(PlanForAssertion(assertion_id=node.id, steps=[step]))

    return {"plans": [p.model_dump() for p in all_plans]}, state.update(
        plans=[p.model_dump() for p in all_plans]
    )


@action(reads=["assertion_tree", "plans", "tool_catalog"], writes=["report_json"])
def render_report(state: State, *, planning_model: Optional[str] = None) -> Tuple[Dict[str, Any], State]:
    """
    Assemble a VerificationPlanReport and emit JSON for downstream/UI.
    """
    report = VerificationPlanReport(
        planning_model=planning_model,
        source_summary=None,  # optional: a separate summarize action can fill this
        assertions=[AssertionNode.model_validate(state["assertion_tree"])],
        plans=[PlanForAssertion.model_validate(p) for p in state.get("plans", [])],
        tool_catalog_used=[ToolRef.model_validate(t) for t in state.get("tool_catalog", [])],
        notes="Stage-1 planning only; no tools executed.",
    )
    report_json = report.model_dump_json(indent=2)
    return {"report_json": report_json}, state.update(report_json=report_json)


# ------------------------- Agent -------------------------

class AssertionVerifier(BasePTAAgent):
    """
    Burr-first PTA agent:
      - perceive(): trivial (accepts text)
      - think(): builds + runs a Burr Application (extract -> plan -> render)
      - act(): returns the final VerificationPlanReport JSON string

    No tool execution in Stage-1; only planning. Tool policy/allow-list should be enforced by the
    service/factory that constructs the ToolBus and passes a curated tool_catalog here.
    """

    def __init__(
        self,
        llm: Any,
        *,
        name: str = "assertion-verifier",
        tool_catalog: Optional[List[ToolRef]] = None,
        planner: PlannerSettings = PlannerSettings(),
    ) -> None:
        self.llm = llm
        self.name = name
        self.planner = planner
        self.tool_catalog: List[ToolRef] = tool_catalog or []
        self.state: Dict[str, Any] = {}

    # ---------- PTA: Perceive / Think / Act ----------

    async def perceive(self, user_input: str) -> Dict[str, Any]:
        """Return an initial perception/state payload."""
        return {"input_text": user_input}

    async def think(self, perceived: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build and run the Burr Application:
          perceive_extract_assertions -> plan_steps_for_assertions -> render_report
        """
        # Prepare schemas for structured output prompts
        assertion_schema = AssertionNode.model_json_schema()
        plan_schema = PlanForAssertion.model_json_schema()
        catalog_dicts = [t.model_dump() for t in self.tool_catalog]

        # Build Burr app
        app = (
            ApplicationBuilder()
            .with_actions(
                perceive_extract_assertions.bind(
                    llm=self.llm,
                    assertion_schema=assertion_schema,
                ),
                plan_steps_for_assertions.bind(
                    llm=self.llm,
                    plan_schema=plan_schema,
                    max_steps_per_assertion=self.planner.max_steps_per_assertion,
                ),
                render_report.bind(planning_model=getattr(self.llm, "deployment", None) or getattr(self.llm, "model", None)),
            )
            .with_transitions(
                ("perceive_extract_assertions", "plan_steps_for_assertions", default),
                ("plan_steps_for_assertions", "render_report", default),
            )
            .with_state(  # initial state
                **{
                    "input_text": perceived["input_text"],
                    "tool_catalog": catalog_dicts,
                }
            )
            .with_entrypoint("perceive_extract_assertions")
            .build()
        )

        # Run — single pass to 'render_report'
        # Returns (last_action_name, outputs_dict, final_state)
        last_action, outputs, final_state = app.run()

        self.state["burr_last_action"] = last_action
        self.state["burr_outputs"] = outputs
        self.state["burr_state"] = dict(final_state)  # State → dict
        return {"report_json": final_state["report_json"]}

    async def act(self, thought: Dict[str, Any]) -> str:
        """Return the report JSON (string)."""
        report_json: str = thought["report_json"]
        # Keep a parsed copy in self.state for downstream services/UIs
        self.state["assertion_report"] = json.loads(report_json)
        return report_json

    # Optional: override run() if your PTA base doesn't already do this sequencing.
    async def run(self, user_input: str) -> str:
        perceived = await self.perceive(user_input)
        thought = await self.think(perceived)
        return await self.act(thought)


# ------------------------- local helpers -------------------------

def _flatten_nodes(node: AssertionNode) -> List[AssertionNode]:
    out = [node]
    for c in node.children:
        out.extend(_flatten_nodes(c))
    return out


def _guess_tool(statement: str, catalog: List[Dict[str, Any]]) -> Optional[str]:
    s = statement.lower()
    def find_prefix(prefixes: List[str]) -> Optional[str]:
        for t in catalog:
            name = t.get("name", "").lower()
            for p in prefixes:
                if name.startswith(p):
                    return t["name"]
        return None

    if any(k in s for k in ["revenue", "profit", "q1", "q2", "q3", "q4", "$", "usd", "m", "billion", "kpi"]):
        return find_prefix(["finance", "sql", "text2sql"])
    if any(k in s for k in ["policy", "guideline", "manual", "procedure"]):
        return find_prefix(["vector", "kb.search", "semantic"])
    if any(k in s for k in ["customer", "account", "client", "crm"]):
        return find_prefix(["crm", "customer", "id.lookup"])
    if any(k in s for k in ["date", "time", "deadline", "quarter", "year"]):
        return find_prefix(["calendar", "temporal", "reporting"])
    return catalog[0]["name"] if catalog else None
