import argparse

from power_shapes.functions import run_compute_wasserstein_distance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and save the Wasserstein distance matrix."
    )
    parser.add_argument("--date-min", default="2026-01-01", help="First date to include.")
    parser.add_argument("--date-max", default="2026-06-01", help="Last date to include.")
    parser.add_argument("--order", type=int, default=1, help="Wasserstein distance order.")
    parser.add_argument(
        "--internal-p",
        type=float,
        default=2,
        help="Ground metric parameter used to compare persistence diagram points.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Approximation parameter for the Wasserstein distance.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--output",
        default="data/wasserstein_distance_matrix.csv",
        help="Path of the output CSV file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    distance_matrix_df = run_compute_wasserstein_distance(
        date_min=args.date_min,
        date_max=args.date_max,
        order=args.order,
        internal_p=args.internal_p,
        delta=args.delta,
        n_jobs=args.n_jobs,
    )

    distance_matrix_df.to_csv(args.output, index=True)


if __name__ == "__main__":
    main()