# create a script to create a dictionary that has the first column in cog-20.def.tab as the key and the second column as the value

import csv
import json

def main():
    with open("cog-20.def.tab", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        cog_to_label = {row[0]: row[1][0] for row in reader}
    
    with open("cog_to_label.json", "w") as f:
        json.dump(cog_to_label, f)
        
main()