from __future__ import annotations

import json
from pathlib import Path

import yaml

from work_agent.recipes.csv_errors import summarize_error_csv


def build_eval_repair_task(*, repo_root: Path, errors_csv: Path, profile_path: Path) -> str:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    summary = summarize_error_csv(errors_csv, aliases=profile.get("csv_aliases", {}))

    failed_doc_list = repo_root / ".work_agent_failed_docs.csv"
    failed_doc_list.write_text("doc_id\n" + "\n".join(summary["doc_ids"]) + "\n", encoding="utf-8")

    return f"""
You are repairing a Forms Agent use case using an error-gradient CSV.

Goal:
- Inspect the CSV summary.
- Pick the highest-value failure cluster.
- Inspect relevant code/prompts.
- Propose and apply a minimal patch only if evidence is sufficient.
- Rerun the failed docs/evals using the commands below.
- Run regression/unit tests if available.
- Finish with a report.

Repository:
{repo_root}

Errors CSV:
{errors_csv}

Generated failed-doc list:
{failed_doc_list}

CSV summary:
{json.dumps(summary, indent=2, ensure_ascii=False)}

Preferred file areas:
{json.dumps(profile.get("repair", {}).get("prefer_files", []), indent=2)}

Available command templates:
{json.dumps(profile.get("commands", {}), indent=2)}

Constraints:
- Do not hardcode document IDs.
- Do not hardcode expected values from the CSV.
- Change at most a few localized files.
- Prefer parser/normalizer/validator fixes over prompt bloat.
- After patching, run the failed-doc eval command and summarize before/after behavior.
"""
