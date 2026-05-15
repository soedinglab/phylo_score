use lazy_static::lazy_static;
use seq_io::fasta::Record;
use std::{collections::HashMap, path::Path, vec};

lazy_static! {
    static ref NUC_MAPPING: HashMap<u8, [f64;4]> = {
        let mut m = HashMap::new();
        m.insert(b'A', [1.0, 0.0, 0.0, 0.0]);
        m.insert(b'C', [0.0, 1.0, 0.0, 0.0]);
        m.insert(b'G', [0.0, 0.0, 1.0, 0.0]);
        m.insert(b'T', [0.0, 0.0, 0.0, 1.0]);
        m.insert(b'U', [0.0, 0.0, 0.0, 1.0]);
        m.insert(b'-', [1.0, 1.0, 1.0, 1.0]);
        m.insert(b'.', [1.0, 1.0, 1.0, 1.0]);
        m.insert(b'N', [0.25, 0.25, 0.25, 0.25]);
        m.insert(b'R', [0.5, 0.0, 0.5, 0.0]);
        m.insert(b'Y', [0.0, 0.5, 0.0, 0.5]);
        m.insert(b'S', [0.0, 0.5, 0.5, 0.0]);
        m.insert(b'W', [0.5, 0.0, 0.0, 0.5]);
        m.insert(b'K', [0.0, 0.0, 0.5, 0.5]);
        m.insert(b'M', [0.5, 0.5, 0.0, 0.0]);
        m.insert(b'B', [0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0]);
        m.insert(b'D', [1.0/3.0, 0.0, 1.0/3.0, 1.0/3.0]);
        m.insert(b'H', [1.0/3.0, 1.0/3.0, 0.0, 1.0/3.0]);
        m.insert(b'V', [1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0]);
        m
    };
}

pub fn read_alignment(fasta_file: &Path) -> HashMap<String, Vec<u8>> {
    let mut alignment = seq_io::fasta::Reader::new(
        std::fs::File::open(fasta_file).expect("Could not open fasta file"),
    );
    let mut sequences = HashMap::new();
    while let Some(result) = alignment.next() {
        let record = result.expect("Error reading fasta record");
        let seq: Vec<u8> = record.owned_seq();
        sequences.insert(record.id().unwrap().to_string(), seq);
    }
    sequences
}

pub fn process_newick_alignment(
    newick: &str,
    sequences: &HashMap<String, Vec<u8>>,
    pairs: impl Iterator<Item = (usize, usize)>,
) -> (phylo_grad::FelsensteinTree<16>, Vec<Vec<phylo_grad::nalgebra::SVector<f64, 16>>>) {
    let tree = phylotree::tree::Tree::from_newick(newick).unwrap();

    let level_order = tree.levelorder(&tree.get_root().unwrap()).unwrap();

    let mut parents = vec![i32::MAX; level_order.len()];
    let mut distances = vec![f64::NAN; level_order.len()];

    let num_leaves = tree.n_leaves();

    // idx_mapping[tree_idx] = new_idx
    let mut idx_mapping = vec![i32::MAX; level_order.len()];

    let mut leaf_idx = 0usize;
    let mut internal_idx = level_order.len() - 1;

    for node_idx in &level_order {
        let node = tree.get(node_idx).unwrap();
        if node.is_tip() {
            idx_mapping[*node_idx] = leaf_idx as i32;
            parents[leaf_idx] = idx_mapping[node.parent.unwrap()];
            distances[leaf_idx] = node.parent_edge.unwrap();
            leaf_idx += 1;
        } else {
            idx_mapping[*node_idx] = internal_idx as i32;
            if let Some(parent_idx) = node.parent {
                parents[internal_idx] = idx_mapping[parent_idx];
                distances[internal_idx] = node.parent_edge.unwrap();
            } else {
                parents[internal_idx] = -1;
            }
            internal_idx -= 1;
        }
    }

    let mut leaf_pl = vec![];

    for (i,j) in pairs {
        let mut leaf_pl_col = vec![];
        for node_idx in level_order.iter() {
            let node = tree.get(&node_idx).unwrap();
            if node.is_tip() {
                let new_idx = idx_mapping[*node_idx];
                assert!(new_idx < num_leaves as i32);
                let seq = sequences.get(node.name.as_ref().unwrap()).unwrap();
                let n_i = seq[i];
                let n_j = seq[j];
                if n_i == b'-' || n_j == b'-' || n_i == b'.' || n_j == b'.' {
                    // Any of the sequences has a gap, so we set the pll to all 1s, essently ignoring this position in the likelihood calculation
                    leaf_pl_col.push(phylo_grad::nalgebra::SVector::<f64, 16>::from_element(1.0));
                } else {
                    let pl_i = NUC_MAPPING.get(&n_i).unwrap_or_else(|| panic!("Invalid nucleotide: {}", n_i as char));
                    let pl_j = NUC_MAPPING.get(&n_j).unwrap_or_else(|| panic!("Invalid nucleotide: {}", n_j as char));
                    let mut pl = phylo_grad::nalgebra::SVector::<f64, 16>::zeros();
                    for a in 0..4 {
                        for b in 0..4 {
                            pl[a * 4 + b] = pl_i[a] * pl_j[b];
                        }
                    }
                    leaf_pl_col.push(pl);
                }
            }
        }
        leaf_pl.push(leaf_pl_col);
    }

    let felsenstein = phylo_grad::FelsensteinTree::<16>::new(&parents, &distances);
    (felsenstein, leaf_pl)
}

pub fn read_seq_names(fasta_file: &Path) -> Vec<String> {
    let mut alignment = seq_io::fasta::Reader::new(
        std::fs::File::open(fasta_file).expect("Could not open fasta file"),
    );
    let mut names = vec![];
    while let Some(result) = alignment.next() {
        let record = result.expect("Error reading fasta record");
        names.push(record.id().unwrap().to_string());
    }
    names
}

pub fn write_fasta(
    output_path: &Path,
    order: &[String],
    null_alignment: &HashMap<String, Vec<u8>>,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    for id in order {
        let seq = null_alignment
            .get(id)
            .unwrap_or_else(|| panic!("Null alignment missing sequence for '{id}'"));
        writer.write_all(b">")?;
        writer.write_all(id.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.write_all(seq)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()
}