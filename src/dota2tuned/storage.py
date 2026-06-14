from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


def utc_run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_parquet(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    schema: dict[str, pl.DataType] | None = None,
) -> int:
    data = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        pl.DataFrame(schema=schema).write_parquet(path)
        return 0
    if schema:
        data = [{column: row.get(column) for column in schema} for row in data]
    pl.DataFrame(data, schema=schema, strict=False).write_parquet(path)
    return len(data)


def read_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def connect_duckdb(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def refresh_views(duckdb_path: Path, parquet_dir: Path) -> None:
    conn = connect_duckdb(duckdb_path)
    try:
        for parquet in parquet_dir.glob("*.parquet"):
            view_name = _quote_identifier(parquet.stem)
            parquet_path = _quote_literal(str(parquet))
            conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet({parquet_path})"
            )
    finally:
        conn.close()
