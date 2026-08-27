#!/usr/bin/env python3
"""Execute-only Telemanom-ESA Mission1 full target-channel run (reuse model.h5).

Skips training. Use after a successful full-channel train that failed on execute,
or when re-scoring with the same hyperparameters.

  conda activate timeeval
  cd TimeEval-algorithms && docker build -t registry.gitlab.hpi.de/akita/i/telemanom_esa ./telemanom_esa

  python mission1_telemanom_esa_full_execute_only.py \\
    --model-dir results/<timestamp>/Telemanom-ESA/<hash>/ESA-Mission1/3_months/1
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

TARGET_CHANNELS = [
    "channel_12", "channel_13", "channel_14", "channel_15", "channel_16", "channel_17",
    "channel_18", "channel_19", "channel_20", "channel_21", "channel_22", "channel_23",
    "channel_24", "channel_25", "channel_26", "channel_27", "channel_28", "channel_29",
    "channel_30", "channel_31", "channel_32", "channel_33", "channel_34", "channel_35",
    "channel_36", "channel_37", "channel_38", "channel_39", "channel_40", "channel_41",
    "channel_42", "channel_43", "channel_44", "channel_45", "channel_46", "channel_47",
    "channel_48", "channel_49", "channel_50", "channel_51", "channel_52", "channel_57",
    "channel_58", "channel_59", "channel_60", "channel_61", "channel_62", "channel_63",
    "channel_64", "channel_65", "channel_66", "channel_70", "channel_71", "channel_72",
    "channel_73", "channel_74", "channel_75", "channel_76",
]

VALIDATION_SPLITS = {
    "3_months": "2000-03-11",
    "10_months": "2000-09-01",
    "21_months": "2001-07-01",
    "42_months": "2003-04-01",
    "84_months": "2006-10-01",
}

REQUIRED_MODEL_FILES = ("model.h5", "model.h5.means", "model.h5.stds")


def _install_skip_training(pretrained_dir: Path) -> None:
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
        description="Telemanom-ESA Mission1 full target channels EXECUTE ONLY"
    )
    parser.add_argument("--dataset", default="3_months", choices=list(VALIDATION_SPLITS.keys()))
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model.h5, model.h5.means, model.h5.stds",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=None,
        help="Optional Docker memory cap in GiB (omit for host auto limit)",
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
        f"EXECUTE-ONLY Telemanom-ESA FULL on {collection}/{args.dataset} "
        f"(target_channels={len(TARGET_CHANNELS)}, layers=[134, 134], model={pretrained})"
    )

    limits_kwargs = {
        "tasks_per_host": 1,
        "train_timeout": Duration("1h"),
        "execute_timeout": Duration("120h"),
    }
    if args.memory_gb is not None:
        limits_kwargs["task_memory_limit"] = args.memory_gb * GB

    limits = ResourceConstraints(**limits_kwargs)

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
                    "layers": [134, 134],
                    "validation_date_split": split,
                    "target_channels": TARGET_CHANNELS,
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
