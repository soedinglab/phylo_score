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

all_mask = []
all_scores = []

for score, ss in zip(snakemake.input.scores, snakemake.input.ss):
    scores = np.load(score)
    ss = open(ss, 'r').read().strip()
    true_pairs = pairs_from_secondary(ss)
    all_pairs = np.triu_indices(len(ss), k = 1)

    mask = [p in true_pairs for p in zip(all_pairs[0], all_pairs[1])]
    
    all_mask.extend(mask)
    all_scores.extend(-scores['p_values'])

import pickle

with open(snakemake.output[0], 'wb') as f:
    pickle.dump((all_mask, all_scores), f)