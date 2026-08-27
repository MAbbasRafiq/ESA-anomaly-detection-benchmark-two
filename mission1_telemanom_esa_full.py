#!/usr/bin/env python3
"""Run Telemanom-ESA on Mission1 full target-channel setup (paper Table 2 full set).

58 target channels, layers=[134, 134]. input_channels is omitted so Telemanom-ESA
uses all non-anomaly columns in the preprocessed CSV (telemetry + telecommands).

Designed for GPU lab hosts (64 GB RAM recommended). One train horizon at a time.

  cd TimeEval-algorithms
  sudo docker build -t registry.gitlab.hpi.de/akita/i/telemanom_esa ./telemanom_esa

Usage:
  conda activate timeeval
  python mission1_telemanom_esa_full.py
  python mission1_telemanom_esa_full.py --dataset 84_months
  python mission1_telemanom_esa_full.py --memory-gb 48   # optional Docker RAM cap
"""
from pathlib import Path
import argparse
import os

from durations import Duration

from timeeval import TimeEval, DatasetManager, ResourceConstraints
from timeeval.metrics import ESAScores, ChannelAwareFScore, ADTQC
from timeeval.params import FixedParameters
from timeeval.resource_constraints import GB
from timeeval_experiments.algorithms import telemanom_esa

current_dir = os.path.dirname(os.path.realpath(__file__))
data_raw_folder = os.path.abspath(os.path.join(current_dir, "data"))
data_processed_folder = os.path.abspath(os.path.join(data_raw_folder, "preprocessed"))

# Same target list as mission1_experiments.py (58 Target=YES channels)
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


def main():
    parser = argparse.ArgumentParser(
        description="Telemanom-ESA Mission1 full target channels (58 outputs, layers=[134, 134])"
    )
    parser.add_argument(
        "--dataset",
        default="3_months",
        choices=list(VALIDATION_SPLITS.keys()),
        help="Train horizon (default: 3_months for smoke test; use 84_months for paper default split)",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=None,
        help="Optional Docker memory cap in GiB. Omit to use host RAM minus 1 GiB (recommended on lab GPU box).",
    )
    args = parser.parse_args()

    if args.dataset not in VALIDATION_SPLITS:
        raise SystemExit(f"Unknown dataset {args.dataset}")

    dm = DatasetManager(data_processed_folder)
    collection = "ESA-Mission1"
    datasets = dm.select(collection=collection, dataset=args.dataset)
    if not datasets:
        raise SystemExit(
            f"No dataset '{args.dataset}' in {data_processed_folder}. "
            "Generate it with notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py"
        )

    split = VALIDATION_SPLITS[args.dataset]
    print(
        f"Running Telemanom-ESA FULL on {collection}/{args.dataset} "
        f"(target_channels={len(TARGET_CHANNELS)}, layers=[134, 134], "
        f"input=all CSV columns, validation_date_split={split})"
    )

    limits_kwargs = {
        "tasks_per_host": 1,
        "train_timeout": Duration("120h"),
        "execute_timeout": Duration("120h"),
    }
    if args.memory_gb is not None:
        limits_kwargs["task_memory_limit"] = args.memory_gb * GB
        print(f"Docker memory cap: {args.memory_gb} GiB")
    else:
        print("Docker memory: auto (host RAM - 1 GiB)")

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
