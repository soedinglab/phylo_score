use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

// This is a reimplementation of R-Scapes Null alignment generation

const STATE_A: u8 = 1 << 0;
const STATE_C: u8 = 1 << 1;
const STATE_G: u8 = 1 << 2;
const STATE_U: u8 = 1 << 3;
const STATE_GAP: u8 = 1 << 4;
const ALL_STATES: u8 = STATE_A | STATE_C | STATE_G | STATE_U | STATE_GAP;
const OUTPUT_STATES: [u8; 5] = [b'A', b'C', b'G', b'U', b'-'];

#[derive(Debug, Clone)]
struct ParsedTree {
    tree: phylotree::tree::Tree,
    root: usize,
    preorder: Vec<usize>,
    postorder: Vec<usize>,
    leaves: Vec<usize>,
}

pub fn generate_null_alignment(
    newick: &str,
    sequences: &HashMap<String, Vec<u8>>,
    seed: u64,
) -> Result<HashMap<String, Vec<u8>>, String> {
    generate_null_alignment_with_mask(newick, sequences, None, seed)
}

pub fn generate_null_alignment_with_mask(
    newick: &str,
    sequences: &HashMap<String, Vec<u8>>,
    use_columns: Option<&[bool]>,
    seed: u64,
) -> Result<HashMap<String, Vec<u8>>, String> {
    let parsed = parse_tree(newick)?;
    let alignment_length = validate_alignment(&parsed, sequences)?;
    let mask = normalize_mask(use_columns, alignment_length)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let original = reconstruct_ancestral_alignment(&parsed, sequences, &mut rng, alignment_length)?;
    let mut shuffled = shuffle_alignment_columns(&original, &mask, &mut rng);

    for &node_id in parsed.preorder.iter().skip(1) {
        let node = parsed
            .tree
            .get(&node_id)
            .map_err(|_| format!("Missing tree node {node_id}"))?;
        let parent_id = node
            .parent
            .ok_or_else(|| format!("Node {node_id} is missing its parent"))?;
        let shuffled_parent = shuffled[parent_id].clone();
        apply_branch_substitutions(
            &original[parent_id],
            &original[node_id],
            &shuffled_parent,
            &mut shuffled[node_id],
            &mask,
            &mut rng,
        )?;
    }

    let mut null_alignment = HashMap::with_capacity(parsed.leaves.len());
    for &leaf_id in &parsed.leaves {
        let node = parsed
            .tree
            .get(&leaf_id)
            .map_err(|_| format!("Missing leaf node {leaf_id}"))?;
        let name = node
            .name
            .clone()
            .ok_or_else(|| format!("Leaf node {leaf_id} is missing a name"))?;
        null_alignment.insert(name, shuffled[leaf_id].clone());
    }

    Ok(null_alignment)
}

fn parse_tree(newick: &str) -> Result<ParsedTree, String> {
    let tree = phylotree::tree::Tree::from_newick(newick).map_err(|err| err.to_string())?;
    let root = tree
        .get_root()
        .map_err(|_| "Tree does not have a root".to_string())?;

    let mut preorder = Vec::new();
    let mut postorder = Vec::new();
    let mut leaves = Vec::new();
    traverse_tree(&tree, root, &mut preorder, &mut postorder, &mut leaves)?;

    Ok(ParsedTree {
        tree,
        root,
        preorder,
        postorder,
        leaves,
    })
}

fn traverse_tree(
    tree: &phylotree::tree::Tree,
    node_id: usize,
    preorder: &mut Vec<usize>,
    postorder: &mut Vec<usize>,
    leaves: &mut Vec<usize>,
) -> Result<(), String> {
    preorder.push(node_id);
    let node = tree
        .get(&node_id)
        .map_err(|_| format!("Missing tree node {node_id}"))?;
    if node.is_tip() {
        leaves.push(node_id);
    } else {
        for &child_id in &node.children {
            traverse_tree(tree, child_id, preorder, postorder, leaves)?;
        }
    }
    postorder.push(node_id);
    Ok(())
}

fn validate_alignment(
    parsed: &ParsedTree,
    sequences: &HashMap<String, Vec<u8>>,
) -> Result<usize, String> {
    let first_length = sequences
        .values()
        .next()
        .ok_or_else(|| "Alignment is empty".to_string())?
        .len();

    for &leaf_id in &parsed.leaves {
        let node = parsed
            .tree
            .get(&leaf_id)
            .map_err(|_| format!("Missing leaf node {leaf_id}"))?;
        let name = node
            .name
            .as_ref()
            .ok_or_else(|| format!("Leaf node {leaf_id} is missing a name"))?;
        let sequence = sequences
            .get(name)
            .ok_or_else(|| format!("Missing sequence for tree leaf '{name}'"))?;
        if sequence.len() != first_length {
            return Err(format!(
                "Alignment length mismatch for leaf '{name}': expected {first_length}, got {}",
                sequence.len()
            ));
        }
        for &residue in sequence {
            if residue_to_mask(residue) == 0 {
                return Err(format!(
                    "Unsupported residue '{}' in sequence '{name}'",
                    residue as char
                ));
            }
        }
    }

    Ok(first_length)
}

fn normalize_mask(
    use_columns: Option<&[bool]>,
    alignment_length: usize,
) -> Result<Vec<bool>, String> {
    match use_columns {
        Some(mask) => {
            if mask.len() != alignment_length {
                return Err(format!(
                    "Column mask length mismatch: expected {alignment_length}, got {}",
                    mask.len()
                ));
            }
            Ok(mask.to_vec())
        }
        None => Ok(vec![true; alignment_length]),
    }
}

fn reconstruct_ancestral_alignment(
    parsed: &ParsedTree,
    sequences: &HashMap<String, Vec<u8>>,
    rng: &mut StdRng,
    alignment_length: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let node_count = parsed
        .preorder
        .iter()
        .copied()
        .max()
        .map(|x| x + 1)
        .unwrap_or(0);
    let mut reconstructed = vec![vec![b'-'; alignment_length]; node_count];
    let mut state_sets = vec![0_u8; node_count];

    for column in 0..alignment_length {
        let mut weights = [0.0_f64; 5];

        for &leaf_id in &parsed.leaves {
            let leaf = parsed
                .tree
                .get(&leaf_id)
                .map_err(|_| format!("Missing leaf node {leaf_id}"))?;
            let name = leaf
                .name
                .as_ref()
                .ok_or_else(|| format!("Leaf node {leaf_id} is missing a name"))?;
            let residue = sequences
                .get(name)
                .ok_or_else(|| format!("Missing sequence for tree leaf '{name}'"))?[column];
            let mask = residue_to_mask(residue);
            if mask == 0 {
                return Err(format!(
                    "Unsupported residue '{}' in sequence '{name}'",
                    residue as char
                ));
            }
            state_sets[leaf_id] = mask;
            add_mask_weights(mask, &mut weights);
        }

        for &node_id in &parsed.postorder {
            let node = parsed
                .tree
                .get(&node_id)
                .map_err(|_| format!("Missing tree node {node_id}"))?;
            if node.is_tip() {
                continue;
            }

            let mut children = node.children.iter().copied();
            let first_child = children
                .next()
                .ok_or_else(|| format!("Internal node {node_id} has no children"))?;
            let mut combined = state_sets[first_child];
            for child_id in children {
                let child_states = state_sets[child_id];
                let intersection = combined & child_states;
                combined = if intersection != 0 {
                    intersection
                } else {
                    combined | child_states
                };
            }
            state_sets[node_id] = combined;
        }

        let root_state = choose_state(state_sets[parsed.root], &weights, rng)?;
        reconstructed[parsed.root][column] = root_state;
        assign_descendant_states(
            parsed,
            parsed.root,
            column,
            root_state,
            &state_sets,
            &weights,
            rng,
            &mut reconstructed,
        )?;
    }

    Ok(reconstructed)
}

fn assign_descendant_states(
    parsed: &ParsedTree,
    node_id: usize,
    column: usize,
    parent_state: u8,
    state_sets: &[u8],
    weights: &[f64; 5],
    rng: &mut StdRng,
    reconstructed: &mut [Vec<u8>],
) -> Result<(), String> {
    let node = parsed
        .tree
        .get(&node_id)
        .map_err(|_| format!("Missing tree node {node_id}"))?;

    for &child_id in &node.children {
        let child_mask = state_sets[child_id];
        let child_state = if residue_to_mask(parent_state) & child_mask != 0 {
            parent_state
        } else {
            choose_state(child_mask, weights, rng)?
        };
        reconstructed[child_id][column] = child_state;
        assign_descendant_states(
            parsed,
            child_id,
            column,
            child_state,
            state_sets,
            weights,
            rng,
            reconstructed,
        )?;
    }

    Ok(())
}

fn shuffle_alignment_columns(
    alignment: &[Vec<u8>],
    use_columns: &[bool],
    rng: &mut StdRng,
) -> Vec<Vec<u8>> {
    let selected = (0..use_columns.len())
        .filter(|&column| use_columns[column])
        .collect::<Vec<_>>();
    let mut permuted = selected.clone();
    permuted.shuffle(rng);

    let mut shuffled = alignment.to_vec();
    for (sequence_idx, sequence) in shuffled.iter_mut().enumerate() {
        for (&target, &source) in selected.iter().zip(permuted.iter()) {
            sequence[target] = alignment[sequence_idx][source];
        }
    }

    shuffled
}

fn apply_branch_substitutions(
    original_ancestor: &[u8],
    original_descendant: &[u8],
    shuffled_ancestor: &[u8],
    shuffled_descendant: &mut [u8],
    use_columns: &[bool],
    rng: &mut StdRng,
) -> Result<(), String> {
    if original_ancestor.len() != original_descendant.len()
        || original_ancestor.len() != shuffled_ancestor.len()
        || original_ancestor.len() != shuffled_descendant.len()
        || original_ancestor.len() != use_columns.len()
    {
        return Err("Branch sequences have incompatible lengths".to_string());
    }

    shuffled_descendant.copy_from_slice(shuffled_ancestor);

    let mut counts = [[0_usize; 5]; 5];
    for column in 0..original_ancestor.len() {
        if !use_columns[column] {
            continue;
        }
        let old_state = state_index(original_ancestor[column]);
        let new_state = state_index(original_descendant[column]);
        if let (Some(old_state), Some(new_state)) = (old_state, new_state) {
            if old_state != new_state {
                counts[old_state][new_state] += 1;
            }
        }
    }

    for old_state in 0..OUTPUT_STATES.len() {
        let mut positions = (0..shuffled_ancestor.len())
            .filter(|&column| {
                use_columns[column] && shuffled_ancestor[column] == OUTPUT_STATES[old_state]
            })
            .collect::<Vec<_>>();
        positions.shuffle(rng);

        let mut offset = 0;
        for new_state in 0..OUTPUT_STATES.len() {
            let n_changes = counts[old_state][new_state];
            if n_changes == 0 {
                continue;
            }
            if offset + n_changes > positions.len() {
                return Err(format!(
                    "Not enough shuffled positions to apply substitutions {} -> {}",
                    OUTPUT_STATES[old_state] as char, OUTPUT_STATES[new_state] as char
                ));
            }
            for &column in &positions[offset..offset + n_changes] {
                shuffled_descendant[column] = OUTPUT_STATES[new_state];
            }
            offset += n_changes;
        }
    }

    Ok(())
}

fn residue_to_mask(residue: u8) -> u8 {
    match residue.to_ascii_uppercase() {
        b'A' => STATE_A,
        b'C' => STATE_C,
        b'G' => STATE_G,
        b'T' | b'U' => STATE_U,
        b'-' | b'.' => STATE_GAP,
        b'R' => STATE_A | STATE_G,
        b'Y' => STATE_C | STATE_U,
        b'S' => STATE_C | STATE_G,
        b'W' => STATE_A | STATE_U,
        b'K' => STATE_G | STATE_U,
        b'M' => STATE_A | STATE_C,
        b'B' => STATE_C | STATE_G | STATE_U,
        b'D' => STATE_A | STATE_G | STATE_U,
        b'H' => STATE_A | STATE_C | STATE_U,
        b'V' => STATE_A | STATE_C | STATE_G,
        b'N' => ALL_STATES,
        _ => 0,
    }
}

fn add_mask_weights(mask: u8, weights: &mut [f64; 5]) {
    let count = mask.count_ones() as f64;
    if count == 0.0 {
        return;
    }
    for (index, state) in OUTPUT_STATES.iter().enumerate() {
        if residue_to_mask(*state) & mask != 0 {
            weights[index] += 1.0 / count;
        }
    }
}

fn choose_state(mask: u8, weights: &[f64; 5], rng: &mut StdRng) -> Result<u8, String> {
    let mut allowed = Vec::new();
    let mut total = 0.0;

    for (index, &state) in OUTPUT_STATES.iter().enumerate() {
        if residue_to_mask(state) & mask != 0 {
            let weight = if weights[index] > 0.0 {
                weights[index]
            } else {
                1.0
            };
            allowed.push((state, weight));
            total += weight;
        }
    }

    if allowed.is_empty() {
        return Err("Cannot choose a state from an empty Fitch set".to_string());
    }

    let fallback = allowed.last().map(|(state, _)| *state).unwrap_or(b'-');
    let mut draw = rng.r#gen::<f64>() * total;
    for (state, weight) in allowed {
        if draw <= weight {
            return Ok(state);
        }
        draw -= weight;
    }

    Ok(fallback)
}

fn state_index(residue: u8) -> Option<usize> {
    match residue {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'U' => Some(3),
        b'-' => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(sequence: &[u8]) -> [usize; 5] {
        let mut out = [0; 5];
        for &residue in sequence {
            out[state_index(residue).unwrap()] += 1;
        }
        out
    }

    #[test]
    fn branch_substitutions_preserve_descendant_composition() {
        let original_ancestor = b"AACCGGUU--".to_vec();
        let original_descendant = b"ACCCGGU-U-".to_vec();
        let shuffled_ancestor = b"UGCAU-CGA-".to_vec();
        let mut shuffled_descendant = vec![b'-'; original_ancestor.len()];
        let mask = vec![true; original_ancestor.len()];

        apply_branch_substitutions(
            &original_ancestor,
            &original_descendant,
            &shuffled_ancestor,
            &mut shuffled_descendant,
            &mask,
            &mut StdRng::seed_from_u64(7),
        )
        .unwrap();

        assert_eq!(counts(&shuffled_descendant), counts(&original_descendant));
    }

    #[test]
    fn generated_null_alignment_preserves_leaf_composition() {
        let newick = "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);";
        let sequences = HashMap::from([
            ("A".to_string(), b"AACCGGUU--".to_vec()),
            ("B".to_string(), b"AACUGGUU--".to_vec()),
            ("C".to_string(), b"UUCCGGAA--".to_vec()),
            ("D".to_string(), b"UUCCGAAA--".to_vec()),
        ]);

        let shuffled = generate_null_alignment(newick, &sequences, 19).unwrap();

        for (name, sequence) in &sequences {
            assert_eq!(counts(sequence), counts(shuffled.get(name).unwrap()));
            assert_eq!(sequence.len(), shuffled.get(name).unwrap().len());
        }
    }
}
