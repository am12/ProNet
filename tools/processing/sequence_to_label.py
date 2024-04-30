# parse the cog-20.fa file and create a dictionary with the sequence as the key and the label as the value using the protein_id_to_label.json file because every sequence has a protein_id

import json
import os
import gzip
import csv

def main():
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
        
main()
        