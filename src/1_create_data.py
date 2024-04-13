import numpy as np
from pyfaidx import Fasta
import os
import csv
from progress.bar import Bar

protein_dict = {}
data_dir = '../data/'
out_dir = '../results/1/'
os.makedirs(os.path.dirname(out_dir), exist_ok=True)

# extract sequence name, sequence, and label for every sequence

def get_mappings():
    # COG ID to label
    print('Mapping COG IDs to function...')
    with open(f"{data_dir}cog-20.def.tab", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        cog_to_label = {row[0]: row[1][0] for row in reader} # only gets the primary label
    print(f'\t[INFO] {len(cog_to_label)} COGs found.')
    print(f'\t[INFO] {len(set(cog_to_label.values()))} functions found.')

    # GenBank protein ID to COG ID
    print('Mapping Protein IDs to COG IDs...')
    with open(f'{data_dir}cog-20.cog.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        prot_to_cog = {row[2]: row[6] for row in reader}
    print(f'\t[INFO] {len(prot_to_cog)} proteins found.')

    # load fasta file into pyfaidx Fasta object
    print('Loading protein sequences...')
    proteins = Fasta(f'{data_dir}cog-20.fa', sequence_always_upper=True, key_function=lambda s: s[:-2] + '.' + s[-1])
    print(f'\t[INFO] {len(proteins.keys())} proteins loaded.')

    return cog_to_label, prot_to_cog, proteins

def create_data(): 
    
    cog_to_label, prot_to_cog, proteins = get_mappings()

    with open(f'{out_dir}dataset.csv', 'w') as f:
        
        # write all information to master csv file 
        count = 0
        pbar = Bar('Writing dataset...', max=len(proteins.keys()))
        for protein in proteins:
            prot_id = str(protein.name)
            try:
                cog_id = prot_to_cog[prot_id]
            except KeyError:
                print(f'\t[ERR] prot_id {prot_id} cannot be found. Skipping...')
                pbar.next()
                continue
            try: 
                label = cog_to_label[cog_id]
            except KeyError:
                print(f'\t[ERR] cog_id {cog_id} cannot be found. Skipping...')
                pbar.next()
                continue
            f.write(f'{prot_id},{cog_id},{label},{protein}\n')
            pbar.next()
            count += 1
        pbar.finish()
        print(f'\t[INFO] {count} proteins written.')

if __name__ == '__main__':
    create_data()