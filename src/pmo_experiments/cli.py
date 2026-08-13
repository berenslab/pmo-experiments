"""Root command group for the ``pmo_experiments`` CLI."""

import os
from typing import Any

import click

from pmo_experiments.util.logging_util import setup_logging

# Chain of default output locations across pipeline stages
DEFAULT_EVALUATION_RESULTS_DIR = "evaluation_results"
DEFAULT_AGGREGATION_OUTPUT_DIR = os.path.join(
    DEFAULT_EVALUATION_RESULTS_DIR, "aggregated_scores"
)
DEFAULT_AUROC_CSV_PATH = os.path.join(
    DEFAULT_AGGREGATION_OUTPUT_DIR, "final_test_scores_auroc.csv"
)


PIPELINE_EPILOG = (
    "\b\n"
    "1. configs generate\n"
    "2. train <config path>.yaml \n"
    "3. eval run --eval-split test\n"
    "4. eval run --eval-split val\n"
    "5. eval aggregate\n"
    "6. plots auroc / plots winrate"
)


class GlobalOptionsMixin:
    """
    Mixin adding a "Global options" help section.

    Documents options declared on the root ``cli`` group (e.g. ``--log-level``) in the
    ``--help`` output of every subcommand/subgroup, under a "Global options" section
    (mirroring tools like ``uv``), since click does not otherwise show a parent
    group's options in a subcommand's own help text.
    """

    def format_options(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Render normal options, then append the "Global options" section."""
        super().format_options(  # pyright: ignore[reportAttributeAccessIssue]
            ctx, formatter
        )

        # Skip on the root group itself: its own options already list --log-level.
        own_params = self.get_params(ctx)  # pyright: ignore[reportAttributeAccessIssue]
        if any(getattr(p, "name", None) == "log_level" for p in own_params):
            return

        root_ctx = ctx.find_root()
        log_level_option = next(
            p
            for p in root_ctx.command.params
            if getattr(p, "name", None) == "log_level"
        )
        record = log_level_option.get_help_record(root_ctx)
        if record is None:
            return

        # Match the column width click just used for the "Options" section above
        own_records = [
            r for p in own_params if (r := p.get_help_record(ctx)) is not None
        ]
        own_col_max = max((len(term) for term, _ in own_records), default=30)
        col_width = min(own_col_max, 30)

        term, help_text = record
        with formatter.section("Global options"):
            formatter.write_dl([(term.ljust(col_width), help_text)], col_max=col_width)


class PipelineSectionGroup(GlobalOptionsMixin, click.Group):
    """A ``click.Group`` that renders its epilog as a titled "Pipeline" section."""

    def format_epilog(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, _ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Write `self.epilog` inside a titled "Pipeline" help section."""
        if self.epilog:
            with formatter.section("Pipeline"):
                formatter.write_text(self.epilog)

    def group(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Create the subgroup as this class, so "Global options" cascades to it."""
        kwargs.setdefault("cls", PipelineSectionGroup)
        return super().group(*args, **kwargs)

    def command(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Create the subcommand as `GlobalOptionsCommand` for its help section."""
        kwargs.setdefault("cls", GlobalOptionsCommand)
        return super().command(*args, **kwargs)


class GlobalOptionsCommand(GlobalOptionsMixin, click.Command):
    """A ``click.Command`` that documents the root group's options in its own help."""


@click.group(
    cls=PipelineSectionGroup,
    epilog=PIPELINE_EPILOG,
    context_settings={"max_content_width": 120},
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    metavar="LEVEL",
    help="Logging verbosity. WARNING and above also disable tqdm progress bars. "
    "[debug|info|warning|error|critical]",
)
def cli(log_level: str) -> None:
    """pmo_experiments: CLI for the pmo-experiments pipeline."""
    setup_logging(None, level=log_level)


@cli.command("train", help="Train a CLIP model.")
@click.option(
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the configuration YAML file.",
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a checkpoint (.pt) to resume training from.",
)
@click.argument("overrides", nargs=-1)
def train(
    config_path: str, resume_from: str | None, overrides: tuple[str, ...]
) -> None:
    """
    Train a CLIP model.

    Args:
        config_path (str): Path to the configuration YAML file.
        resume_from (str | None): Path to a checkpoint (.pt) to resume training from.
        overrides (tuple[str, ...]): Configuration overrides in the format key=value.

    """
    from pmo_experiments.scripts.train_model import (  # noqa: PLC0415
        run_training,
    )

    run_training(config_path, resume_from, overrides)


@cli.group("configs")
def configs() -> None:
    """Generate and inspect the model/seed/dataset sweep config tree."""


@configs.command(
    "generate",
    help="Generate the model x seed x dataset sweep config tree.",
)
@click.option(
    "--output-root",
    default="generated_configs",
    show_default=True,
    help="Directory the generated config tree is written to.",
)
@click.option(
    "--model-folder-base",
    default="clip_models",
    show_default=True,
    help="Base name for OUTPUT.SAVE_DIR; the seed is appended as f'{base}_{seed}'.",
)
@click.option(
    "--models",
    multiple=True,
    help="Model catalog keys to generate configs for (see `configs list-models`). "
    "Repeatable. If omitted, generates configs for the default models.",
)
@click.option(
    "--seeds",
    type=int,
    multiple=True,
    help="Seed(s) to sweep. Repeatable. If omitted, generates configs for the "
    "default seeds.",
)
@click.option(
    "--datasets",
    multiple=True,
    help="Registered dataset names to sweep over (see `configs list-datasets`). "
    "Repeatable. If omitted, generates configs for the default datasets.",
)
@click.option(
    "--wandb-project-name",
    default="pmo-experiments",
    show_default=True,
    help="Base name for the wandb project. Used to generate the run name.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print planned paths without writing any files.",
)
def configs_generate(
    output_root: str,
    model_folder_base: str,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    datasets: tuple[str, ...],
    wandb_project_name: str,
    dry_run: bool,
) -> None:
    """
    Generate the model x seed x dataset sweep config tree.

    Args:
        output_root (str): Directory the generated config tree is written to.
        model_folder_base (str): Base name for OUTPUT.SAVE_DIR; the seed is appended
            as f'{base}_{seed}'.
        models (tuple[str, ...]): Model catalog keys to generate configs for. If
            empty, generates configs for the default models.
        seeds (tuple[int, ...]): Seed(s) to sweep. If empty, generates configs for the
            default seeds.
        datasets (tuple[str, ...]): Registered dataset names to sweep over. If empty,
            generates configs for the default datasets.
        wandb_project_name (str): Base name for the wandb project. Used to generate
            the run name.
        dry_run (bool): Print planned paths without writing any files.

    """
    from pmo_experiments.scripts.generate_configs import (  # noqa: PLC0415
        generate_all_configs,
    )

    all_configs = generate_all_configs(
        output_root,
        model_folder_base,
        list(models) if len(models) > 0 else None,
        list(datasets) if len(datasets) > 0 else None,
        list(seeds) if len(seeds) > 0 else None,
        save=not dry_run,
        wandb_project_name=wandb_project_name,
    )

    if dry_run:
        print(f"Would generate {len(all_configs)} configs.")
    else:
        print(f"Generated {len(all_configs)} configs!")


@configs.command("list-models")
def configs_list_models() -> None:
    """List the known model catalog keys."""
    from pmo_experiments.scripts.generate_configs import (  # noqa: PLC0415
        list_models,
    )

    list_models()


@configs.command("list-datasets")
def configs_list_datasets() -> None:
    """List the registered dataset names."""
    from pmo_experiments.scripts.generate_configs import (  # noqa: PLC0415
        list_datasets,
    )

    list_datasets()


@cli.group("eval")
def eval_group() -> None:
    """Evaluate trained CLIP models and aggregate results."""


@eval_group.command(
    "run",
    help=(
        "Evaluate trained CLIP models across datasets.\n\n"
        "Discovers all models at a given epoch from one or more model-base "
        "folders and runs the full evaluation on each (model, eval_dataset) "
        "pair. Already-completed steps are skipped."
    ),
)
@click.option(
    "--model-base-folders",
    multiple=True,
    help="Folders to search for trained model runs (each is globbed for "
    "'<run_prefix>*' subfolders). Repeatable. If omitted, uses every "
    "'clip_models_*' folder found in the current directory.",
)
@click.option(
    "--baseline-folder",
    default="clip_baseline_models",
    show_default=True,
    help="Folder containing baseline (non-fine-tuned) model folders.",
)
@click.option(
    "--epoch",
    type=int,
    default=50,
    show_default=True,
    help="Epoch checkpoint to evaluate.",
)
@click.option(
    "--seeds",
    multiple=True,
    help="Only include models with these training seeds. Repeatable.",
)
@click.option(
    "--models",
    multiple=True,
    help="Only include these model architectures: vit_b_16 vit_b_32 vit_l_14 rn50 "
    "rn50x4. Repeatable.",
)
@click.option(
    "--pretrains",
    multiple=True,
    help="Only include these pretraining sources: openai scratch biomed_clip. "
    "Repeatable.",
)
@click.option(
    "--train-datasets",
    multiple=True,
    help="Only include models trained on these datasets. Repeatable.",
)
@click.option(
    "--eval-datasets",
    multiple=True,
    help="Datasets to evaluate on. Repeatable. If omitted, evaluates on all datasets.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_EVALUATION_RESULTS_DIR,
    show_default=True,
    help="Root output directory.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    show_default=True,
    help="Batch size for embedding extraction.",
)
@click.option(
    "--workers",
    type=int,
    default=16,
    show_default=True,
    help="Number of dataloader workers.",
)
@click.option(
    "--device", default="cuda", show_default=True, help="Device to load models onto."
)
@click.option(
    "--include-baselines",
    is_flag=True,
    help="Also evaluate the pre-trained baseline models from --baseline-folder.",
)
@click.option(
    "--eval-split",
    default="test",
    show_default=True,
    help="Which split to evaluate on.",
)
@click.option(
    "--force-rescore",
    is_flag=True,
    help="Recompute every step even if reports exist, overwriting them and writing "
    "the score .npz files. Use to backfill runs produced before raw scores were "
    "persisted.",
)
@click.option(
    "--skip-baseline-autocreate",
    is_flag=True,
    help="Skip auto-creating baseline model folders if they don't exist. If not "
    "set, baseline folders will be created on-the-fly.",
)
@click.option(
    "--run-prefix",
    default="pmo-experiments",
    show_default=True,
    help="Prefix for the run name. Usually constructed from the wandb project name "
    "and metadata.",
)
def eval_run(
    model_base_folders: tuple[str, ...],
    baseline_folder: str,
    epoch: int,
    seeds: tuple[str, ...],
    models: tuple[str, ...],
    pretrains: tuple[str, ...],
    train_datasets: tuple[str, ...],
    eval_datasets: tuple[str, ...],
    output_dir: str,
    batch_size: int,
    workers: int,
    device: str,
    include_baselines: bool,
    eval_split: str,
    force_rescore: bool,
    skip_baseline_autocreate: bool,
    run_prefix: str,
) -> None:
    """
    Evaluate trained CLIP models across datasets.

    Discovers all models at a given epoch from one or more model-base folders and
    runs the full evaluation on each (model, eval_dataset) pair. Already-completed
    steps are skipped.

    Args:
        model_base_folders (tuple[str, ...]): Folders to search for trained model
            runs. If empty, uses every 'clip_models_*' folder in the current
            directory.
        baseline_folder (str): Folder containing baseline (non-fine-tuned) model
            folders.
        epoch (int): Epoch checkpoint to evaluate.
        seeds (tuple[str, ...]): Only include models with these training seeds. If
            empty, no filtering is applied.
        models (tuple[str, ...]): Only include these model architectures. If empty,
            no filtering is applied.
        pretrains (tuple[str, ...]): Only include these pretraining sources. If
            empty, no filtering is applied.
        train_datasets (tuple[str, ...]): Only include models trained on these
            datasets. If empty, no filtering is applied.
        eval_datasets (tuple[str, ...]): Datasets to evaluate on. If empty, evaluates
            on all datasets.
        output_dir (str): Root output directory.
        batch_size (int): Batch size for embedding extraction.
        workers (int): Number of dataloader workers.
        device (str): Device to load models onto.
        include_baselines (bool): Also evaluate the pre-trained baseline models from
            `baseline_folder`.
        eval_split (str): Which split to evaluate on.
        force_rescore (bool): Recompute every step even if reports exist,
            overwriting them and writing the score .npz files.
        skip_baseline_autocreate (bool): Skip auto-creating baseline model folders if
            they don't exist.
        run_prefix (str): Prefix for the run name. Usually constructed from the wandb
            project name and metadata.

    """
    from pmo_experiments.scripts.run_evaluation import (  # noqa: PLC0415
        run_evaluation_sweep,
    )

    run_evaluation_sweep(
        model_base_folders=list(model_base_folders)
        if len(model_base_folders) > 0
        else None,
        baseline_folder=baseline_folder,
        epoch=epoch,
        seeds=list(seeds) if len(seeds) > 0 else None,
        models=list(models) if len(models) > 0 else None,
        pretrains=list(pretrains) if len(pretrains) > 0 else None,
        train_datasets=list(train_datasets) if len(train_datasets) > 0 else None,
        eval_datasets=list(eval_datasets) if len(eval_datasets) > 0 else None,
        output_dir=output_dir,
        batch_size=batch_size,
        workers=workers,
        device=device,
        include_baselines=include_baselines,
        eval_split=eval_split,
        force_rescore=force_rescore,
        skip_baseline_autocreate=skip_baseline_autocreate,
        run_prefix=run_prefix,
    )


@eval_group.command(
    "aggregate",
    help="Aggregate evaluation scores from multiple models and datasets into "
    "CSV files.",
)
@click.option(
    "--evaluation-folder-path",
    type=click.Path(exists=True, file_okay=False),
    default=DEFAULT_EVALUATION_RESULTS_DIR,
    show_default=True,
    help="Output folder of `eval run`.",
)
@click.option(
    "--number-of-threads",
    type=int,
    default=6,
    show_default=True,
    help="Number of threads used for processing.",
)
@click.option(
    "--aggregation-output-path",
    type=click.Path(file_okay=False),
    default=None,
    help="Path to save the aggregated csv files to. If None creates 'aggregated_scores'"
    " inside the evaluation folder.",
)
@click.option(
    "--metric",
    "metric_list",
    multiple=True,
    help="Metric to aggregate. Repeatable. If empty uses balanced_accuracy, auroc and "
    "auprc.",
)
@click.option(
    "--run-prefix",
    default="pmo-experiments",
    show_default=True,
    help="Prefix for the run name. Usually constructed from the wandb project name "
    "and metadata.",
)
def aggregate(
    evaluation_folder_path: str,
    number_of_threads: int,
    aggregation_output_path: str | None,
    metric_list: tuple[str, ...],
    run_prefix: str,
) -> None:
    """
    Aggregate evaluation scores from multiple models and datasets into CSV files.

    Args:
        evaluation_folder_path (str): Output folder of `eval run`.
        number_of_threads (int): Number of threads used for processing.
        aggregation_output_path (str | None): Path to save the aggregated csv files
            to. If not given creates folder 'aggregated_scores' in the evaluation
            folder.
        metric_list (tuple[str, ...]): Metrics to aggregate. If empty defaults to
            'balanced_accuracy', 'auroc' and 'auprc'.
        run_prefix (str): Prefix for the run name. Usually constructed from the wandb
            project name and metadata.

    """
    from pmo_experiments.evaluation.aggregate_scores import (  # noqa: PLC0415
        create_aggregated_csv,
    )

    create_aggregated_csv(
        evaluation_folder_path,
        number_of_threads=number_of_threads,
        aggregation_output_path=aggregation_output_path,
        metric_list=list(metric_list) if len(metric_list) > 0 else None,
        run_prefix=run_prefix,
    )


@cli.group("plots")
def plots() -> None:
    """Generate result figures from aggregated evaluation score CSVs."""


@plots.command(
    "auroc",
    help="Build the general-case AUROC figure (PMO vs. competitor corpora).",
)
@click.option(
    "--csv-path",
    type=click.Path(exists=True, dir_okay=False),
    default=DEFAULT_AUROC_CSV_PATH,
    show_default=True,
    help="Path to the per-target AUROC CSV, as written by `eval aggregate`.",
)
@click.option(
    "--output-dir",
    default="figures",
    show_default=True,
    help="Directory the SVG/PNG/PDF outputs are saved into.",
)
@click.option(
    "--dpi",
    type=int,
    default=400,
    show_default=True,
    help="Resolution used when saving.",
)
def auroc(csv_path: str, output_dir: str, dpi: int) -> None:
    """
    Build the general-case AUROC figure (PMO vs. competitor corpora).

    Args:
        csv_path (str): Path to the per-target AUROC CSV, as written by `scores
            aggregate`.
        output_dir (str): Directory the SVG/PNG/PDF outputs are saved into.
        dpi (int): Resolution used when saving.

    """
    from pmo_experiments.plots.figure_auroc import (  # noqa: PLC0415
        plot_general_case_figure,
    )

    plot_general_case_figure(csv_path, output_dir=output_dir, dpi=dpi)


@plots.command(
    "winrate",
    help="Build the PMO win-rate figure across eval protocols and architectures.",
)
@click.option(
    "--csv-path",
    type=click.Path(exists=True, dir_okay=False),
    default=DEFAULT_AUROC_CSV_PATH,
    show_default=True,
    help="Path to the per-target AUROC CSV, as written by `eval aggregate`.",
)
@click.option(
    "--output-dir",
    default="figures",
    show_default=True,
    help="Directory the SVG/PNG/PDF outputs are saved into.",
)
@click.option(
    "--dpi",
    type=int,
    default=400,
    show_default=True,
    help="Resolution used when saving.",
)
def winrate(csv_path: str, output_dir: str, dpi: int) -> None:
    """
    Build the PMO win-rate figure across eval protocols and architectures.

    Args:
        csv_path (str): Path to the per-target AUROC CSV, as written by `scores
            aggregate`.
        output_dir (str): Directory the SVG/PNG/PDF outputs are saved into.
        dpi (int): Resolution used when saving.

    """
    from pmo_experiments.plots.figure_winrate_auroc import (  # noqa: PLC0415
        plot_winrate_figure,
    )

    plot_winrate_figure(csv_path, output_dir=output_dir, dpi=dpi)


@plots.command(
    "tables",
    help="Generate the PMO vs. competitor corpora tables.",
)
@click.option(
    "--csv-path",
    type=click.Path(exists=True, dir_okay=False),
    default=DEFAULT_AUROC_CSV_PATH,
    show_default=True,
    help="Path to the per-target AUROC CSV, as written by `eval aggregate`.",
)
@click.option(
    "--output-dir",
    default="eval_tables",
    show_default=True,
    help="Directory the CSV outputs are saved into.",
)
def tables(csv_path: str, output_path: str) -> None:
    """
    Generate the PMO vs. competitor corpora tables.

    Args:
        csv_path (str): Path to the per-target AUROC CSV, as written by `scores
            aggregate`.
        output_path (str): Directory the CSV outputs are saved into.

    """
    from pmo_experiments.plots.generate_tables import (  # noqa: PLC0415
        generate_tables,
    )

    generate_tables(csv_path, output_path)


if __name__ == "__main__":
    cli()
