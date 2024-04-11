# iterate over every COG.tsv.gz in the fasta directory, and access the first column of the file and create a dictionary with the first column as the key and the COG label as the value

import csv
import json
import os
import gzip

def main():
    
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


main()