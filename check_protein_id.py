# create a script to take in one protein id and check i with protein_id_to_label.json

import json
import sys

def main():
    with open("protein_id_to_label.json", "r") as f:
        protein_id_to_label = json.load(f)
    
    protein_id = sys.argv[1]
    print(protein_id_to_label.get(protein_id))
    
main()