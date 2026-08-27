#!/usr/bin/env python3
"""Run Telemanom-ESA on Mission1 lightweight subset (6 channels, layers=[80, 80]).

Designed for ~16GB RAM hosts: only the subset config, one train horizon at a time,
single Docker task. Rebuild the telemanom_esa image after the usecols memory patch:

  cd TimeEval-algorithms
  sudo docker build -t registry.gitlab.hpi.de/akita/i/telemanom_esa ./telemanom_esa

Usage:
  conda activate timeeval
  python mission1_telemanom_esa_subset.py
  python mission1_telemanom_esa_subset.py --dataset 10_months
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

# Same lightweight subset as mission1_experiments.py
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


def main():
    parser = argparse.ArgumentParser(description="Telemanom-ESA Mission1 subset (6 channels)")
    parser.add_argument(
        "--dataset",
        default="3_months",
        choices=list(VALIDATION_SPLITS.keys()),
        help="Train horizon (default: 3_months — smallest / best for low RAM)",
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
    print(f"Running Telemanom-ESA subset on {collection}/{args.dataset} "
          f"(channels={SUBSET_CHANNELS}, layers=[80, 80], validation_date_split={split})")

    # Physical RAM cap; on CPU hosts DockerAdapter also enables swap (see docker.py).
    limits = ResourceConstraints(
        tasks_per_host=1,
        task_memory_limit=10 * GB,
        train_timeout=Duration("120h"),
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
