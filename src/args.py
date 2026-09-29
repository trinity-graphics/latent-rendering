import argparse


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Specify a config file path, defaults to `configs/default.yaml`.",
    )
    parser.add_argument(
        "-o",
        "--train_only",
        action="store_true",
        help="Only performs optimization and outputs .pt files, skipping testing.",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--test_only",
        action="store_true",
        help="Only runs testing, skipping optimization.  Assumes that pre-trained .pt files can be found under the `out_dir` specified in the config.",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--test_log",
        action="store_false",
        help="Disables logging for test runs too.",
        default=True,
    )
    parser.add_argument(
        "-r",
        "--refiner_training",
        action="store_true",
        help="Only runs training for refiner, skipping scene optimization.  Assumes that pre-trained .pt files can be found under the `out_dir` specified in the config.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="Benchmarks the pipeline.",
        default=False,
    )

    return parser.parse_known_args()