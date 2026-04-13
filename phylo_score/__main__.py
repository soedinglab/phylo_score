import argparse
import numpy as np

from . import optimize_models, utils

def main():
    parser = argparse.ArgumentParser(description="Compute co-evolution scores using phylogeny.")
    parser.add_argument("--alignment", help="Path to the alignment file.", required=True)
    parser.add_argument("--tree", help="Path to the phylogenetic tree file.", required=True)
    parser.add_argument("--output", help="Path to the output file.", required=True)
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=0, help="Number of threads to use. 0 meaning all available threads (default: %(default)s).")
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
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        type=str,
        default=None,
        help="Directory to use for caching intermediate results. If not specified we use ./phylo_score_cache.",
    )
    parser.add_argument(
        "--score-type",
        dest="score_type",
        type=str,
        default="optdiff",
        choices=["lldiff", "optdiff"],
        help="Type of score to compute (default: %(default)s).",
    )
    parser.add_argument(
        "--correction",
        dest="correction",
        type=str,
        default="APC",
        choices=["APC", "ASC", "None"],
        help="Type of correction to apply to the scores (default: %(default)s).",
    )
    args = parser.parse_args()
    
    if args.num_threads > 0:
        import os
        os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)

    ind_ll, pair_ll, ind_reg, pair_reg = optimize_models(
        args.tree,
        args.alignment,
        args.lam,
        args.lambda_mi,
        cache_dir=args.cache_dir,
    )

    scores = utils.calc_scores(ind_ll, pair_ll, ind_reg, pair_reg, args.score_type)
    corrected_scores = utils.calc_correction(scores, args.correction)
    p_vals = utils.p_values(corrected_scores)

    # Save the results to the output file npz format
    np.savez_compressed(
        args.output,
        scores=scores,
        corrected_scores=corrected_scores,
        p_values=p_vals,
        ind_ll=ind_ll,
        pair_ll=pair_ll,
        ind_reg=ind_reg,
        pair_reg=pair_reg,
    )
if __name__ == "__main__":
    main()