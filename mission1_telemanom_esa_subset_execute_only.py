#!/usr/bin/env python3
"""Execute-only Telemanom-ESA Mission1 subset run (reuse an existing model.h5).

Skips the ~9h training step. Use after a successful train that OOM'd on execute,
or any run that left model.h5 + means/stds on disk.

  conda activate timeeval
  # rebuild image after modeling/generators memory fixes:
  cd TimeEval-algorithms && docker build -t registry.gitlab.hpi.de/akita/i/telemanom_esa ./telemanom_esa

  python mission1_telemanom_esa_subset_execute_only.py \\
    --model-dir results/2026_08_12_10_58_07/Telemanom-ESA/08bbe6469b2f9103922bc5c975a70497/ESA-Mission1/3_months/1
"""
from pathlib import Path
import argparse
import os
import shutil

from durations import Duration

from timeeval import TimeEval, DatasetManager, ResourceConstraints
from timeeval.core.experiments import Experiment
from timeeval.metrics import ESAScores, ChannelAwareFScore, ADTQC
from timeeval.params import FixedParameters
from timeeval.resource_constraints import GB
from timeeval_experiments.algorithms import telemanom_esa

current_dir = os.path.dirname(os.path.realpath(__file__))
data_raw_folder = os.path.abspath(os.path.join(current_dir, "data"))
data_processed_folder = os.path.abspath(os.path.join(data_raw_folder, "preprocessed"))

SUBSET_CHANNELS = [
    "channel_41",
    "channel_42",
    "channel_43",
    "channel_44",
    "channel_45",
    "channel_46",
]

VALIDATION_SPLITS = {
    "3_months": "2000-03-11",
    "10_months": "2000-09-01",
    "21_months": "2001-07-01",
    "42_months": "2003-04-01",
    "84_months": "2006-10-01",
}

DEFAULT_MODEL_DIR = os.path.join(
    current_dir,
    "results/2026_08_12_10_58_07/Telemanom-ESA/"
    "08bbe6469b2f9103922bc5c975a70497/ESA-Mission1/3_months/1",
)

REQUIRED_MODEL_FILES = ("model.h5", "model.h5.means", "model.h5.stds")


def _install_skip_training(pretrained_dir: Path) -> None:
    """Replace Experiment._perform_training so Docker execute reuses model.h5."""

    def _perform_training_reuse(self) -> dict:
        self.results_path.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_MODEL_FILES:
            src = pretrained_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing pretrained file: {src}")
            dst = self.results_path / name
            shutil.copy2(src, dst)
            print(f"Reused pretrained {name} -> {dst}")
        with (self.results_path / "execution.log").open("a") as logs_file:
            print(
                f"Skipping training; reused model from {pretrained_dir}",
                file=logs_file,
            )
        return {}

    Experiment._perform_training = _perform_training_reuse  # type: ignore[method-assign]


def main():
    parser = argparse.ArgumentParser(
        description="Telemanom-ESA Mission1 subset EXECUTE ONLY (reuse model.h5)"
    )
    parser.add_argument("--dataset", default="3_months", choices=list(VALIDATION_SPLITS.keys()))
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing model.h5, model.h5.means, model.h5.stds",
    )
    args = parser.parse_args()

    pretrained = Path(args.model_dir).resolve()
    for name in REQUIRED_MODEL_FILES:
        if not (pretrained / name).is_file():
            raise SystemExit(f"Missing {name} in {pretrained}")

    _install_skip_training(pretrained)

    dm = DatasetManager(data_processed_folder)
    collection = "ESA-Mission1"
    datasets = dm.select(collection=collection, dataset=args.dataset)
    if not datasets:
        raise SystemExit(f"No dataset '{args.dataset}' in {data_processed_folder}")

    split = VALIDATION_SPLITS[args.dataset]
    print(
        f"EXECUTE-ONLY Telemanom-ESA on {collection}/{args.dataset} "
        f"(channels={SUBSET_CHANNELS}, layers=[80, 80], model={pretrained})"
    )

    # Physical RAM cap + Docker adapter now allows +48GiB swap on CPU hosts
    limits = ResourceConstraints(
        tasks_per_host=1,
        task_memory_limit=11 * GB,
        train_timeout=Duration("1h"),
        execute_timeout=Duration("120h"),
    )

    beta = 0.5
    metrics = [
        ESAScores(betas=beta, select_labels={"Category": ["Rare Event", "Anomaly"]}),
        ESAScores(betas=beta, select_labels={"Category": ["Anomaly"]}),
    ]
    ranking_metrics = [
        ChannelAwareFScore(beta=beta, select_labels={"Category": ["Rare Event", "Anomaly"]}),
        ChannelAwareFScore(beta=beta, select_labels={"Category": ["Anomaly"]}),
        ADTQC(select_labels={"Category": ["Rare Event", "Anomaly"]}),
        ADTQC(select_labels={"Category": ["Anomaly"]}),
    ]

    labels_csv = Path(os.path.join(data_raw_folder, collection, "labels.csv"))
    test_dataset_path = Path(
        os.path.join(
            data_processed_folder,
            "multivariate",
            f"{collection}-semi-supervised",
            "84_months.test.csv",
        )
    )

    algorithms = [
        telemanom_esa(
            params=FixedParameters(
                {
                    "epochs": 1000,
                    "early_stopping_patience": 20,
                    "layers": [80, 80],
                    "validation_date_split": split,
                    "input_channels": SUBSET_CHANNELS,
                    "target_channels": SUBSET_CHANNELS,
                    "threshold_scores": 1,
                    "min_error_value": 0,
                }
            ),
            skip_pull=True,
        )
    ]

    timeeval = TimeEval(
        dm,
        datasets,
        algorithms,
        metrics=metrics,
        ranking_metrics=ranking_metrics,
        resource_constraints=limits,
        labels_csv_path=labels_csv,
        test_dataset_path=test_dataset_path,
    )
    timeeval.run()
    print(timeeval.get_results(aggregated=False))


if __name__ == "__main__":
    main()
