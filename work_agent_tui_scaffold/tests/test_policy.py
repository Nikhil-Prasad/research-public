from pathlib import Path

import pytest

from work_agent.policy import Policy, PolicyError


def policy() -> Policy:
    return Policy.from_yaml(Path(__file__).parents[1] / "configs" / "policy.yaml")


def test_blocks_inline_python():
    with pytest.raises(PolicyError):
        policy().assert_shell_allowed("python -c 'print(1)'")


def test_allows_pytest():
    argv = policy().assert_shell_allowed("python -m pytest -q")
    assert argv[:3] == ["python", "-m", "pytest"]


def test_blocks_git_apply_via_shell():
    with pytest.raises(PolicyError):
        policy().assert_shell_allowed("git apply patch.diff")
