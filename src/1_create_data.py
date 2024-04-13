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

    # load fasta file into pyfaidx Fasta object
    print('Loading protein sequences...')
    proteins = Fasta(f'{data_dir}cog-20.fa', sequence_always_upper=True, key_function=lambda x: x.replace('_', '.'))

    print(proteins.keys())
    
    # pbar = Bar('Writing')
    # with open

get_labels()
    
    
    
    

def prot_id_to_label():
    id_to_label = {}
    
    with open("cog_to_label.json", "r") as f:
        cog_to_label = json.load(f)
    
    for i in range(1, 5951):
        num = str(i).zfill(4)
        filename = "fasta/COG" + num + ".tsv.gz"
        if os.path.exists(filename):
            with gzip.open(filename, "rt") as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    id = row[0]
                    label = cog_to_label.get("COG" + num)
                    if label is not None:
                        print("COG" + num + ": ", id, label)
                        id_to_label[id] = label
                    else:
                        print("Label not found.")
        else:
            print("File does not exist.")
            
    with open("protein_id_to_label.json", "w") as f:
        json.dump(id_to_label, f)
        print("ID TO LABEL TEST")
        print(id_to_label["ABX12048.1"])

def seq_to_label():
    with open("protein_id_to_label.json", "r") as f:
        protein_id_to_label = json.load(f)
    
    sequence_to_label = {}
    
    # parse the cog-20.fa file 2 lines at a time
    
    with open("cog-20.fa", "r") as f:
        for line in f:
            if line.startswith(">"):
                protein_id = ((line[1:].strip()).split())[0]
                # replace the second to last character with a period
                protein_id = protein_id[:-2] + "." + protein_id[-1]
                print(protein_id)
                label = protein_id_to_label.get(protein_id)
                if label is not None:
                    sequence = next(f).strip()
                    # print(sequence, label)
                    sequence_to_label[sequence] = label
                else:
                    print("Label not found")
                    
    with open("sequence_to_label.json", "w") as f:
        json.dump(sequence_to_label, f)


def create_data():

    os.chdir(input_dir)
    

    pass


if __name__ == '__main__':
    create_data()