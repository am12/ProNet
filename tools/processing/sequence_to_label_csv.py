# create a script that reads the sequence_to_label.json file and creates a sequence_to_label.csv file with 2 columns, sequence and label

import json
import csv

def main():
    with open("sequence_to_label.json", "r") as f:
        sequence_to_label = json.load(f)
    
    with open("sequence_to_label.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        for sequence, label in sequence_to_label.items():
            writer.writerow([sequence, label])
            
main()
