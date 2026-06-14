from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from dota2tuned.config import get_settings
from dota2tuned.evaluate import load_predictor_metrics
from dota2tuned.features import build_feature_summary
from dota2tuned.finetune import (
    launch_hf_job,
    upload_sft_dataset,
    validate_hf_jobs_access,
    write_train_script,
)
from dota2tuned.ingest import IngestCoordinator
from dota2tuned.normalize import normalize_all
from dota2tuned.rag import build_index
from dota2tuned.sft import create_sft_examples
from dota2tuned.smoke import has_failures, run_smoke_checks
from dota2tuned.storage import refresh_views
from dota2tuned.train_predictor import train_predictor
from dota2tuned.ui.gradio_app import build_app

app = typer.Typer(no_args_is_help=True)


@app.command()
def health() -> None:
    settings = get_settings()
    typer.echo(f"raw_data_dir={settings.raw_data_dir}")
    typer.echo(f"parquet_dir={settings.parquet_dir}")
    typer.echo(f"duckdb_path={settings.duckdb_path}")
    typer.echo(f"hf_space_id={settings.hf_space_id}")
    typer.echo(f"base_model_id={settings.base_model_id}")
    typer.echo(f"modal_enabled={settings.modal_enabled}")


@app.command()
def smoke(
    live: Annotated[
        bool,
        typer.Option(help="Run minimal live checks against HF, OpenDota, STRATZ, Steam, Valve."),
    ] = False,
) -> None:
    settings = get_settings()
    results = run_smoke_checks(settings, live=live)
    typer.echo(json.dumps(results, indent=2))
    if has_failures(results):
        raise typer.Exit(1)


@app.command()
def ingest(
    pro_matches: Annotated[int, typer.Option(help="Number of recent pro match summaries.")] = 100,
    public_matches: Annotated[int, typer.Option(help="Number of public match summaries.")] = 100,
    enrich_limit: Annotated[
        int, typer.Option(help="Number of matches to enrich via /matches.")
    ] = 20,
    stratz_limit: Annotated[
        int, typer.Option(help="Number of enriched matches to request from STRATZ.")
    ] = 0,
    patch_count: Annotated[int, typer.Option(help="Recent major patch note count to ingest.")] = 4,
    league_limit: Annotated[int, typer.Option(help="Recent pro league IDs to backfill.")] = 3,
) -> None:
    settings = get_settings()
    coordinator = IngestCoordinator(settings)
    counts = {}
    counts.update(coordinator.ingest_reference())
    counts.update(coordinator.ingest_patches(max_patches=patch_count))
    counts.update(
        coordinator.ingest_matches(
            pro_matches=pro_matches,
            public_matches=public_matches,
            enrich_limit=enrich_limit,
            stratz_limit=stratz_limit,
            league_limit=league_limit,
        )
    )
    typer.echo(counts)


@app.command()
def normalize() -> None:
    settings = get_settings()
    counts = normalize_all(settings.raw_data_dir, settings.parquet_dir)
    refresh_views(settings.duckdb_path, settings.parquet_dir)
    typer.echo(counts)


@app.command()
def features() -> None:
    settings = get_settings()
    typer.echo(build_feature_summary(settings.parquet_dir, settings.duckdb_path))


@app.command("build-rag")
def build_rag() -> None:
    settings = get_settings()
    count = build_index(settings.parquet_dir, settings.rag_dir)
    typer.echo({"documents": count, "index": str(settings.rag_dir / "tfidf.joblib")})


@app.command("train-predictor")
def train_predictor_command() -> None:
    settings = get_settings()
    typer.echo(train_predictor(settings.parquet_dir, settings.model_dir))


@app.command("make-sft")
def make_sft(
    output: Annotated[Path, typer.Option(help="Output JSONL path.")] = Path(
        "data/models/sft_examples.jsonl"
    ),
    limit: Annotated[int, typer.Option(help="Max examples.")] = 100,
) -> None:
    settings = get_settings()
    count = create_sft_examples(settings.parquet_dir, settings.rag_dir, output, limit=limit)
    typer.echo({"examples": count, "output": str(output)})


@app.command()
def finetune(
    dataset_path: Annotated[Path, typer.Option(help="Local JSONL SFT dataset.")] = Path(
        "data/models/sft_examples.jsonl"
    ),
    launch_job: Annotated[bool, typer.Option(help="Launch on Hugging Face Jobs.")] = False,
) -> None:
    settings = get_settings()
    if launch_job:
        try:
            preflight = validate_hf_jobs_access(settings)
        except RuntimeError as exc:
            typer.echo(f"Preflight failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo({"preflight": preflight})
        dataset_source = upload_sft_dataset(settings, dataset_path)
        script = write_train_script(settings, dataset_source)
        typer.echo(launch_hf_job(settings, script))
    else:
        script = write_train_script(settings, dataset_path)
        typer.echo({"script": str(script), "launch": "rerun with --launch-job"})


@app.command()
def eval() -> None:
    settings = get_settings()
    typer.echo(load_predictor_metrics(settings.model_dir))


@app.command()
def serve() -> None:
    demo = build_app()
    demo.launch()


@app.command("modal-deploy")
def modal_deploy() -> None:
    settings = get_settings()
    if not settings.modal_enabled:
        typer.echo("Set MODAL_ENABLED=1 before deploying Modal backend functions.")
        raise typer.Exit(1)
    typer.echo("Run: modal deploy src/dota2tuned/modal_backend.py")
