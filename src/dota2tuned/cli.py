from __future__ import annotations

import json
import os
import subprocess
import sys
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

app = typer.Typer(no_args_is_help=True)


def _modal_env(settings) -> dict[str, str]:
    env = os.environ.copy()
    if settings.modal_token_id:
        env["MODAL_TOKEN_ID"] = settings.modal_token_id
    if settings.modal_token_secret:
        env["MODAL_TOKEN_SECRET"] = settings.modal_token_secret
    return env


def _require_modal(settings) -> None:
    if not settings.modal_enabled:
        typer.echo("Set MODAL_ENABLED=1 before using Modal commands.", err=True)
        raise typer.Exit(1)
    if not settings.modal_token_id or not settings.modal_token_secret:
        typer.echo("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required.", err=True)
        raise typer.Exit(1)
    os.environ.update(_modal_env(settings))


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


@app.command("targeted-ingest")
def targeted_ingest(
    threshold: Annotated[
        int, typer.Option(help="Hero sample threshold that defines high confidence.")
    ] = 500,
    hero_limit: Annotated[int, typer.Option(help="Maximum under-sampled heroes to target.")] = 64,
    matches_per_hero: Annotated[
        int, typer.Option(help="Maximum candidate details to fetch for each targeted hero.")
    ] = 700,
    max_new_details: Annotated[
        int, typer.Option(help="Maximum new match details to append in this run.")
    ] = 5000,
    page_size: Annotated[int, typer.Option(help="Explorer rows per targeted query page.")] = 100,
) -> None:
    settings = get_settings()
    coordinator = IngestCoordinator(settings)
    counts = coordinator.ingest_targeted_hero_matches(
        threshold=threshold,
        hero_limit=hero_limit,
        matches_per_hero=matches_per_hero,
        max_new_details=max_new_details,
        page_size=page_size,
    )
    typer.echo(json.dumps(counts, indent=2))


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
    from dota2tuned.ui.runtime_hooks import (
        close_idle_main_event_loop,
        gradio_launch_runtime_kwargs,
        install_quiet_unraisablehook,
    )

    install_quiet_unraisablehook()

    import gradio as gr

    from dota2tuned.ui.gradio_app import (
        APP_CSS,
        APP_HEAD,
        build_app,
        launch_app_kwargs,
    )

    demo = build_app()
    close_idle_main_event_loop()
    # Load JS is attached inside build_app via demo.load(); launch(js=) does not
    # run on page load in Gradio 6.18. app_kwargs wires the critical-head middleware
    # that prevents the bare-element flash before the loader. ssr_mode=False keeps
    # rendering client-side (HF auto-enables SSR, which bypasses the loader/middleware).
    demo.launch(
        css=APP_CSS,
        head=APP_HEAD,
        theme=gr.themes.Soft(),
        ssr_mode=False,
        **gradio_launch_runtime_kwargs(),
        app_kwargs=launch_app_kwargs(),
    )


@app.command("modal-deploy")
def modal_deploy() -> None:
    settings = get_settings()
    _require_modal(settings)
    command = [
        sys.executable,
        "-m",
        "modal",
        "deploy",
        "-m",
        "dota2tuned.modal_backend",
        "--name",
        settings.modal_app_name,
    ]
    result = subprocess.run(command, env=_modal_env(settings), check=False)
    raise typer.Exit(result.returncode)


@app.command("modal-smoke")
def modal_smoke() -> None:
    settings = get_settings()
    _require_modal(settings)
    try:
        import modal
    except ImportError as exc:
        typer.echo("Install Modal dependencies with `uv sync --extra modal`.", err=True)
        raise typer.Exit(1) from exc

    smoke_fn = modal.Function.from_name(settings.modal_app_name, "remote_smoke")
    typer.echo(json.dumps(smoke_fn.remote(), indent=2))


@app.command("modal-train")
def modal_train(
    dataset_source: Annotated[
        str,
        typer.Option(help="Local-in-Modal JSONL path or Hub dataset id for SFT data."),
    ] = "data/models/sft_examples.jsonl",
    wait: Annotated[bool, typer.Option(help="Wait for training to finish.")] = False,
) -> None:
    settings = get_settings()
    _require_modal(settings)
    if not settings.hf_token:
        typer.echo("HF_TOKEN is required so Modal training can push the adapter.", err=True)
        raise typer.Exit(1)
    try:
        import modal
    except ImportError as exc:
        typer.echo("Install Modal dependencies with `uv sync --extra modal`.", err=True)
        raise typer.Exit(1) from exc

    if dataset_source == "data/models/sft_examples.jsonl":
        dataset_source = "/app/data/models/sft_examples.jsonl"
    train_fn = modal.Function.from_name(settings.modal_app_name, "train_sft")
    if wait:
        typer.echo(json.dumps(train_fn.remote(dataset_source), indent=2))
        return

    call = train_fn.spawn(dataset_source)
    typer.echo(
        json.dumps(
            {
                "status": "submitted",
                "app": settings.modal_app_name,
                "function": "train_sft",
                "function_call_id": getattr(call, "object_id", str(call)),
            },
            indent=2,
        )
    )


@app.command("modal-ask")
def modal_ask(
    question: Annotated[str, typer.Argument(help="Question for the fine-tuned adapter.")],
    context: Annotated[
        str,
        typer.Option(help="Optional evidence/context to pass to the adapter."),
    ] = "",
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 384,
) -> None:
    settings = get_settings()
    _require_modal(settings)
    try:
        import modal
    except ImportError as exc:
        typer.echo("Install Modal dependencies with `uv sync --extra modal`.", err=True)
        raise typer.Exit(1) from exc

    generate_fn = modal.Function.from_name(settings.modal_app_name, "generate_answer")
    typer.echo(json.dumps(generate_fn.remote(question, context, max_new_tokens), indent=2))
