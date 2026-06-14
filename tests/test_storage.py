from pathlib import Path

from dota2tuned.storage import connect_duckdb, refresh_views, write_parquet


def test_refresh_views_creates_duckdb_views(tmp_path: Path):
    parquet_dir = tmp_path / "parquet"
    duckdb_path = tmp_path / "db.duckdb"
    write_parquet(parquet_dir / "example.parquet", [{"id": 1, "name": "ok"}])

    refresh_views(duckdb_path, parquet_dir)

    conn = connect_duckdb(duckdb_path)
    try:
        assert conn.execute("select count(*) from example").fetchone()[0] == 1
    finally:
        conn.close()
