from __future__ import annotations

import argparse
from pathlib import Path

from work_agent.approvals import ApprovalBroker
from work_agent.driver import AgentDriver
from work_agent.llm_client import make_llm_client
from work_agent.policy import Policy
from work_agent.recipes.eval_repair import build_eval_repair_task
from work_agent.run_store import RunStore


def make_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("work-agent")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--policy", default="configs/policy.yaml")
    parser.add_argument("--mode", default="patch-assist")
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock LLM")
    return parser


def main() -> None:
    parser = argparse.ArgumentParser("work-agent")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run")
    run.add_argument("--repo", required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--policy", default="configs/policy.yaml")
    run.add_argument("--mode", default="patch-assist")
    run.add_argument("--mock", action="store_true")

    repair = sub.add_parser("repair")
    repair.add_argument("--repo", required=True)
    repair.add_argument("--errors", required=True)
    repair.add_argument("--profile", default="configs/profiles/forms_agent.yaml")
    repair.add_argument("--policy", default="configs/policy.yaml")
    repair.add_argument("--mode", default="patch-assist")
    repair.add_argument("--mock", action="store_true")

    tui = sub.add_parser("tui")
    tui.add_argument("--repo", default=".")
    tui.add_argument("--task", default=None)
    tui.add_argument("--policy", default="configs/policy.yaml")
    tui.add_argument("--mode", default="patch-assist")
    tui.add_argument("--mock", action="store_true")

    tui_repair = sub.add_parser("tui-repair")
    tui_repair.add_argument("--repo", required=True)
    tui_repair.add_argument("--errors", required=True)
    tui_repair.add_argument("--profile", default="configs/profiles/forms_agent.yaml")
    tui_repair.add_argument("--policy", default="configs/policy.yaml")
    tui_repair.add_argument("--mode", default="patch-assist")
    tui_repair.add_argument("--mock", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        args.command = "tui"
        args.repo = "."
        args.task = None
        args.policy = "configs/policy.yaml"
        args.mode = "patch-assist"
        args.mock = False

    policy = Policy.from_yaml(args.policy)
    repo_root = Path(args.repo).resolve()
    llm = make_llm_client(mock=getattr(args, "mock", False))
    run_store = RunStore()

    if args.command == "run":
        driver = AgentDriver(
            repo_root=repo_root,
            policy=policy,
            llm=llm,
            run_store=run_store,
            approvals=ApprovalBroker(policy),
        )
        final = driver.run(task=args.task, mode=args.mode)
        print(final)
        print(f"\nArtifacts: {run_store.path}")
        return

    if args.command == "repair":
        task = build_eval_repair_task(repo_root=repo_root, errors_csv=Path(args.errors), profile_path=Path(args.profile))
        driver = AgentDriver(
            repo_root=repo_root,
            policy=policy,
            llm=llm,
            run_store=run_store,
            approvals=ApprovalBroker(policy),
        )
        final = driver.run(task=task, mode=args.mode)
        print(final)
        print(f"\nArtifacts: {run_store.path}")
        return

    if args.command in {"tui", "tui-repair"}:
        if args.command == "tui":
            task = args.task
        else:
            task = build_eval_repair_task(repo_root=repo_root, errors_csv=Path(args.errors), profile_path=Path(args.profile))

        from work_agent.tui.app import WorkAgentTui

        app = WorkAgentTui(
            repo_root=repo_root,
            task=task,
            mode=args.mode,
            policy=policy,
            llm=llm,
            run_store=run_store,
        )
        app.run()
        print(f"Artifacts: {run_store.path}")
        return


if __name__ == "__main__":
    main()
