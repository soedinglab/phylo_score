import argparse
import numpy as np
from pathlib import Path

from . import optimize_models, utils, _phylo_score

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
    parser.add_argument(
        "--p_values",
        dest="p_values",
        type=str,
        default="null_alignment",
        choices=["null_alignment", "fast"],
        help="Type of p_value calculation. Either null_alignment or fast, (default: %(default)s).",
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
    
    if args.p_values == "null_alignment":
        fasta_path = Path(args.alignment)
        null_fasta = fasta_path.parent / (fasta_path.stem + "_null.fasta")
        _phylo_score.null_alignment(str(fasta_path), args.tree, str(null_fasta), 0)
        null_ind_ll, null_pair_ll, null_ind_reg, null_pair_reg = optimize_models(
            args.tree,
            null_fasta,
            args.lam,
            args.lambda_mi,
            cache_dir=args.cache_dir,
        )
        null_scores = utils.calc_scores(null_ind_ll, null_pair_ll, null_ind_reg, null_pair_reg, args.score_type)
        null_corrected_scores = utils.calc_correction(null_scores, args.correction)
        

    scores = utils.calc_scores(ind_ll, pair_ll, ind_reg, pair_reg, args.score_type)
    corrected_scores = utils.calc_correction(scores, args.correction)
    
    if args.p_values == "null_alignment":
        null_p_vals = utils.p_values_with_null(corrected_scores, null_corrected_scores)
    else:
        null_p_vals = None
    
    p_vals = utils.p_values(corrected_scores)

    # Save the results to the output file npz format
    np.savez_compressed(
        args.output,
        uncorrected_scores=scores,
        corrected_scores=corrected_scores,
        fast_p_values=p_vals,
        null_p_values=null_p_vals,
        ind_ll=ind_ll,
        pair_ll=pair_ll,
        ind_reg=ind_reg,
        pair_reg=pair_reg,
    )

    csv_output = Path(args.output).with_suffix(".csv")
    utils.write_pair_scores_csv(
        csv_output,
        scores=scores,
        corrected_scores=corrected_scores,
        fast_p_values=p_vals,
        null_p_values=null_p_vals,
        ind_ll=ind_ll,
        pair_ll=pair_ll,
        ind_reg=ind_reg,
        pair_reg=pair_reg,
    )
    
    
if __name__ == "__main__":
    main()