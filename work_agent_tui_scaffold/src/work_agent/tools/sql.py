from __future__ import annotations

import os
from typing import Any

from work_agent.contracts import SqlResult
from work_agent.policy import Policy


def run_sql(*, query: str, policy: Policy) -> SqlResult:
    policy.assert_sql_allowed(query)

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install optional SQL deps: pip install 'work-agent[sql]'") from exc

    max_rows = policy.data["sql"].get("max_rows", 200)

    conninfo = {
        "host": os.environ["POSTGRES_HOST"],
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }

    with psycopg.connect(**conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN READ ONLY")
            cur.execute(query)

            columns = [desc.name for desc in cur.description]
            raw_rows = cur.fetchmany(max_rows)
            rows: list[dict[str, Any]] = [dict(zip(columns, row)) for row in raw_rows]

            cur.execute("ROLLBACK")

    return SqlResult(columns=columns, rows=rows, row_count=len(rows))
