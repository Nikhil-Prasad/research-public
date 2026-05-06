from __future__ import annotations

import os
from abc import ABC, abstractmethod


from work_agent.contracts import AgentAction
from work_agent.prompt import SYSTEM_PROMPT


class BaseLLMClient(ABC):
    @abstractmethod
    def next_action(self, task: str, state_summary: str) -> AgentAction:
        raise NotImplementedError


class AzureOpenAIResponsesClient(BaseLLMClient):
    """Azure OpenAI v1-compatible Responses client.

    Env vars:
      AZURE_OPENAI_BASE_URL=https://...openai.azure.com/openai/v1/
      AZURE_OPENAI_API_KEY=...
      AZURE_OPENAI_DEPLOYMENT=your deployment name
    """

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install core dependency: pip install openai") from exc

        self.client = OpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
        )
        self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def next_action(self, task: str, state_summary: str) -> AgentAction:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Task:\n"
                        f"{task}\n\n"
                        "Current state and observations:\n"
                        f"{state_summary}\n"
                    ),
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "AgentAction",
                    "strict": True,
                    "schema": AgentAction.model_json_schema(),
                }
            },
        )

        return AgentAction.model_validate_json(response.output_text)


class MockLLMClient(BaseLLMClient):
    """A deterministic local smoke-test client.

    It does not modify files and does not call any external API.
    """

    def __init__(self) -> None:
        self.step = 0

    def next_action(self, task: str, state_summary: str) -> AgentAction:
        self.step += 1
        if self.step == 1:
            return AgentAction(
                action="tool_call",
                reason="Smoke test: list repo root.",
                tool_name="list_dir",
                tool_args={"path": "."},
            )
        if self.step == 2:
            return AgentAction(
                action="tool_call",
                reason="Smoke test: check git status.",
                tool_name="git_status",
                tool_args={},
            )
        return AgentAction(
            action="finish",
            reason="Mock client completed smoke-test actions.",
            final_answer="Mock run complete. The driver, tool registry, and event stream are working.",
        )


def make_llm_client(mock: bool = False) -> BaseLLMClient:
    if mock or os.environ.get("WORK_AGENT_MOCK") == "1":
        return MockLLMClient()
    return AzureOpenAIResponsesClient()
