import argparse
from . import _phylo_score
import numpy as np
import scipy

def main():
    parser = argparse.ArgumentParser(description="Compute co-evolution scores using phylogeny.")
    parser.add_argument("--alignment", help="Path to the alignment file.", required=True)
    parser.add_argument("--tree", help="Path to the phylogenetic tree file.", required=True)
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=1.0,
        help="lambda parameter (default: %(default)s).",
    )
    parser.add_argument(
        "--lambda-mi",
        dest="lambda_mi",
        type=float,
        default=1.0,
        help="lambda_MI parameter (default: %(default)s).",
    )
    args = parser.parse_args()

    _phylo_score.optimize_models(args.tree, args.alignment, args.lam, args.lambda_mi)

if __name__ == "__main__":
    main()