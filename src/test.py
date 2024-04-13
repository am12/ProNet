import numpy as np
from pyfaidx import Fasta
import os
import csv
from progress.bar import Bar

protein_dict = {}
data_dir = '../data/'
out_dir = '../results/'

# extract sequence name, sequence, and label for every sequence

def get_labels():
    # COG ID to label
    print('Retrieving COG IDs...')
    with open(f"{data_dir}cog-20.def.tab", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        cog_to_label = {row[0]: row[1][0] for row in reader} # only gets the primary label
    

    # GenBank protein ID to COG ID
    print('Retrieving Protein IDs...')
    with open(f'{data_dir}cog-20.cog.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        prot_to_cog = {row[2]: row[6] for row in reader}
    
    print(prot_to_cog.items())

    # load fasta file into pyfaidx Fasta object
    print('Loading protein sequences...')
    proteins = Fasta(f'{data_dir}cog-20.fa', sequence_always_upper=True, key_function=lambda x: x[:12] + '.' + x[13:])

    print(proteins.keys())
    
    for protein in proteins:
        print(protein.name)
        print(protein)
    # pbar = Bar('Writing')
    # with open

get_labels()