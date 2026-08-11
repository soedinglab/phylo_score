# PhyloScore

## Installation

### pip
Just install with `pip install phylo_score` and run with `python -m phylo_score`.

We provide pre compiled wheels for Linux on python 3.11-3.14, otherwise a Rust toolchain is needed for installation: https://rust-lang.org/tools/install/

### From source

Just use `pip install .` after cloning the repository. This requires a working python environment and a Rust toolchain.

## Usage
```
usage: python -m phylo_score [-h] --alignment ALIGNMENT --tree TREE --output OUTPUT
                             [--num-threads NUM_THREADS]
                             [--lambda LAM] [--lambda-mi LAMBDA_MI] [--cache-dir CACHE_DIR]
                             [--score-type {lldiff,optdiff}] [--correction {APC,ASC,None}]
                             [--p_values {null_alignment,fast}]

Compute co-evolution scores using phylogeny.

options:
  -h, --help            show this help message and exit
  --alignment ALIGNMENT Path to the alignment file.
  --tree TREE           Path to the phylogenetic tree file (newick format)
  --output OUTPUT       Path to the output file.
  --num-threads NUM_THREADS
                        Number of threads to use. 0 meaning all available threads (default: 0).
  --lambda LAM          lambda parameter (default: 0.5).
  --lambda-mi LAMBDA_MI
                        lambda_MI parameter (default: 0.01).
  --cache-dir CACHE_DIR
                        Directory to use for caching intermediate results. If not specified we use
                        ./phylo_score_cache
  --score-type {lldiff,optdiff}
                        Type of score to compute (default: optdiff).
  --correction {APC,ASC,None}
                        Type of correction to apply to the scores (default: APC).
  --p_values {null_alignment,fast}
                        Type of p_value calculation. Either null_alignment or fast, (default: null_alignment).
```
