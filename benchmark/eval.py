from snakemake.script import snakemake

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def pairs_from_secondary(input_string: str) -> list[tuple[int, int]]:
    """
        Will give a list of pairs of indices that are paired in the secondary structure.
        (i,j): i < j is garanteed
    """
    open_par = ['(', '[', '{', '<', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    close_par = [')', ']', '}', '>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    stacks = {}  # Contains tuple with (is_open, stack)

    for open, close in zip(open_par, close_par):
        stacks[open] = (True, [])
        stacks[close] = (False, stacks[open][1])  # Points to the same stack

    result = []  # List to store pairs of matching indices

    for index, char in enumerate(input_string):
        if char in stacks:
            is_open, stack = stacks[char]
            if is_open:
                stack.append(index)
            else:
                open_index = stack.pop()
                result.append((open_index, index))

    assert (all([not bool(stack) for (_, stack) in stacks.values()]))

    return result

def roc_avarage_precision(score_file : str, ss_file :str):
    scores = np.load(score_file)
    ss = open(ss_file, 'r').read().strip()
    true_pairs = pairs_from_secondary(ss)
    all_pairs = np.triu_indices(len(ss), k = 1)

    mask = [p in true_pairs for p in zip(all_pairs[0], all_pairs[1])]

    if "scores" in scores:
        scores = scores['scores']
    else:
        scores = scores['corrected_scores']

    roc = roc_auc_score(mask, scores)
    ap = average_precision_score(mask, scores)

    return roc, ap

def analyse_score_file_name(score_file : str):
    """
        Given a score file name, extract the dataset, the regularization, the score type and the correction type.
    """
    import re
    pattern = r"data/([^/]+)/reg([^_]+)_mireg([^_]+)_score([^_]+)_corr([^\.]+)\.npz"
    match = re.match(pattern, score_file)
    if match:
        dataset = match.group(1)
        reg = match.group(2)
        mireg = match.group(3)
        score_type = match.group(4)
        corr_type = match.group(5)
        return dataset, reg, mireg, score_type, corr_type
    else:
        raise ValueError(f"Score file name {score_file} does not match the expected pattern.")
    
def analyse_rscape_file_name(rscape_file : str):
    """
        Given a rscape file name, extract the dataset and the score type.
    """
    import re
    pattern = r"data/([^/]+)/rscape_([^\.]+)\.npz"
    match = re.match(pattern, rscape_file)
    if match:
        dataset = match.group(1)
        score = match.group(2)
        return dataset, score
    else:
        raise ValueError(f"Rscape file name {rscape_file} does not match the expected pattern.")

def analyse_comap_file_name(comap_file : str):
    """
        Given a comap file name, extract the dataset and the score variant.
    """
    import re
    pattern = r"data/([^/]+)/comap_([^\.]+)\.npz"
    match = re.match(pattern, comap_file)
    if match:
        dataset = match.group(1)
        variant = match.group(2)
        return dataset, variant
    else:
        raise ValueError(f"Comap file name {comap_file} does not match the expected pattern.")

all_eval = {}

import itertools

for score, ss in zip(snakemake.input.scores, itertools.cycle(snakemake.input.ss)):
    roc, ap = roc_avarage_precision(score, ss)
    print(f"Score file: {score}, ROC AUC: {roc}, Average Precision: {ap}, ss file: {ss}")
    if "rscape" in str(snakemake.output):
        para = analyse_rscape_file_name(score)
        all_eval[para] = (roc, ap)
    elif "comap" in str(snakemake.output):
        para = analyse_comap_file_name(score)
        all_eval[para] = (roc, ap)
    else:
        para = analyse_score_file_name(score)
        all_eval[para] = (roc, ap)

import pickle

with open(snakemake.output[0], 'wb') as f:
    pickle.dump(all_eval, f)