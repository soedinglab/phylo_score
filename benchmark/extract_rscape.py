import numpy as np
import pandas
from snakemake.script import snakemake

rscape = open(snakemake.input.rscape, 'r')

rscape = pandas.read_csv(rscape, comment='#', delimiter='\t', usecols=[1,2,3,5], names=['left', 'right', 'score', 'pvalue'])
L = rscape['right'].max()
rscape.set_index(['left', 'right'], inplace=True)


pairs = np.triu_indices(L, k = 1)

scores = []
pvalues = []

for pair in zip(pairs[0], pairs[1]):
    scores.append(rscape.at[(pair[0].item() + 1, pair[1].item() + 1), 'score'].item())
    pvalues.append(rscape.at[(pair[0].item() + 1, pair[1].item() + 1), 'pvalue'].item())
np.savez(snakemake.output[0], scores=np.array(scores), p_values=np.array(pvalues))