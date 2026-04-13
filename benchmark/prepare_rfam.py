import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from Bio import Align
import Bio
import sys


input_file = sys.argv[1]
output_fasta = sys.argv[2]
output_ss = sys.argv[3]
output_stockholm = sys.argv[4]

#RFAM uses latin-1 encoding, even though the webserver claims it to be utf-8 (only relevent for the citations and similar in stockholm)
alignment = AlignIO.read(open(input_file, "r", encoding='latin-1'), "stockholm")

def prepare(alignment):
    ss = np.array([ord(c) for c in alignment.column_annotations['secondary_structure']], dtype=np.int8)
    rf = np.array([ord(c) for c in alignment.column_annotations['reference_annotation']], dtype=np.int8)
    
    #We want to ignore all sides with . or ~ in the reference_annotation
    mask = np.logical_and(rf != ord('.'), rf != ord('~'))
        
    new = []
    
    for record in alignment:
        seq = np.array([ord(c) for c in record.seq], dtype=np.int8)
        new_seq = seq[mask].tobytes().decode('utf-8')

        #remove / from the sequence names because this will cause problems with SPOT RNA
        new.append(SeqRecord(Seq(new_seq),record.id.replace('/', '_')))
        
    new = Align.MultipleSeqAlignment(new)
    new.column_annotations['secondary_structure'] = ss[mask].tobytes().decode('utf-8')
    
    return new

new = prepare(alignment)


output = open(output_fasta, 'w')

for req in new:
    output.write('>' + req.id + '\n')
    output.write(str(req.seq) + '\n')

ss_output = open(output_ss, 'w')
ss_output.write(new.column_annotations['secondary_structure'])

AlignIO.write(new, output_stockholm, "stockholm")