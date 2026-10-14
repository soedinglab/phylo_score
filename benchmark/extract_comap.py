"""
    Converts a CoMap pairwise output (.sta) into score files for eval.py.

    Usage: python extract_comap.py comap.sta ss.txt out_corr.npz out_corrAPC.npz out_pvalue.npz
"""
import re
import sys

import numpy as np

import utils

sta_file, ss_file, out_corr, out_apc, out_pvalue = sys.argv[1:]

L = len(open(ss_file, 'r').read().strip())
pairs = utils.all_pairs_numpy(L)
index = {(i, j): k for k, (i, j) in enumerate(zip(pairs[0], pairs[1]))}

corr = np.full(len(index), np.nan)
p_values = np.full(len(index), np.nan)

with open(sta_file, 'r') as f:
    next(f)  # header: Group Stat RCmin PRmin Nmin PValue Nsim
    for line in f:
        fields = line.rstrip('\n').split('\t')
        # Groups are written as [i;j] with 1-based alignment columns
        i, j = map(int, re.match(r"\[(\d+);(\d+)\]", fields[0]).groups())
        k = index[(i - 1, j - 1)]
        corr[k] = float(fields[1])
        p_values[k] = float(fields[5]) if fields[5] not in ('', 'NA') else np.nan

assert not np.isnan(corr).all(), f"No pairs found in {sta_file}"

def fill_missing(scores):
    """
        Pairs without a finite score (e.g. constant columns) are ranked last.
    """
    scores = scores.copy()
    finite = np.isfinite(scores)
    scores[~finite] = scores[finite].min() - 1
    return scores

corr = fill_missing(corr)
pvalue_scores = fill_missing(-p_values)

np.savez(out_corr, scores=corr, p_values=p_values)
np.savez(out_apc, scores=utils.calc_correction(corr, "APC"), p_values=p_values)
np.savez(out_pvalue, scores=pvalue_scores, p_values=p_values)
