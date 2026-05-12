use _phylo_score::null_alignment::generate_null_alignment;
use seq_io::fasta::Record;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("Usage: null_alignment <tree.newick> <input.fasta> <output.fasta> [seed]");
        std::process::exit(1);
    }

    let tree_file = &args[1];
    let input_fasta = &args[2];
    let output_fasta = &args[3];
    let seed = if args.len() == 5 {
        args[4]
            .parse::<u64>()
            .expect("Could not parse seed as u64")
    } else {
        0
    };

    let newick = std::fs::read_to_string(tree_file).expect("Could not read tree file");
    let ordered_records = read_fasta_ordered(Path::new(input_fasta));
    let sequence_map = ordered_records
        .iter()
        .map(|(id, seq)| (id.clone(), seq.clone()))
        .collect::<HashMap<_, _>>();

    let null_alignment = generate_null_alignment(&newick, &sequence_map, seed)
        .expect("Failed to generate null alignment");

    write_fasta(Path::new(output_fasta), &ordered_records, &null_alignment)
        .expect("Could not write output FASTA");
}

fn read_fasta_ordered(path: &Path) -> Vec<(String, Vec<u8>)> {
    let mut reader =
        seq_io::fasta::Reader::new(std::fs::File::open(path).expect("Could not open fasta file"));
    let mut out = Vec::new();
    while let Some(result) = reader.next() {
        let record = result.expect("Error reading fasta record");
        out.push((
            record.id().expect("Missing FASTA record id").to_string(),
            record.owned_seq(),
        ));
    }
    out
}

fn write_fasta(
    output_path: &Path,
    ordered_input: &[(String, Vec<u8>)],
    null_alignment: &HashMap<String, Vec<u8>>,
) -> std::io::Result<()> {
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    for (id, _) in ordered_input {
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