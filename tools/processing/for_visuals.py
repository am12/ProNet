import sys
# create a script to create a dictionary that has the first column in cog-20.def.tab as the key and the second column as the value
import csv
import json
import os
import gzip
import copy

def main():
    with open("protein_id_to_label.json", "r") as f:
        protein_id_to_label = json.load(f)
    
    sequence_to_label = {}
    labels_items = []
    
    i_lines = []
    j_lines = []
    e_lines = []
    g_lines = []
    c_lines = []
    f_lines = []
    o_lines = []
    h_lines = []
    p_lines = []

    # parse the cog-20.fa file 2 lines at a time
    
    with open("cog-20.fa", "r") as f:
        for i in range(2 * 50000):
            static_line = f.readline()
            line = copy.deepcopy(static_line)
            if line.startswith(">"):
                protein_id = ((line[1:].strip()).split())[0]
                # replace the second to last character with a period
                protein_id = protein_id[:-2] + "." + protein_id[-1]
                #print(protein_id)
                label = protein_id_to_label.get(protein_id)
                if label is not None:
                    pre_sequence = next(f)
                    if label == 'I':
                        i_lines.append(static_line)
                        i_lines.append(pre_sequence)
                    elif label == 'J':
                        j_lines.append(static_line)
                        j_lines.append(pre_sequence)
                    elif label == 'E':
                        e_lines.append(static_line)
                        e_lines.append(pre_sequence)
                    elif label == 'G':
                        g_lines.append(static_line)
                        g_lines.append(pre_sequence)
                    elif label == 'C':
                        c_lines.append(static_line)
                        c_lines.append(pre_sequence)
                    elif label == 'F':
                        f_lines.append(static_line)
                        f_lines.append(pre_sequence)
                    elif label == 'O':
                        o_lines.append(static_line)
                        o_lines.append(pre_sequence)
                    elif label == 'H':
                        h_lines.append(static_line)
                        h_lines.append(pre_sequence)
                    elif label == 'P':
                        p_lines.append(static_line)
                        p_lines.append(pre_sequence)
                    sequence = pre_sequence.strip()
                    # print(sequence, label)
                    sequence_to_label[sequence] = label
                    labels_items.append(label)
                #else:
                    #print("Label not found")
    print(set(labels_items))
    # Write to files
    to_write = open('i_lines.fa', 'w')
    for j in i_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('j_lines.fa', 'w')
    for j in j_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('e_lines.fa', 'w')
    for j in e_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('g_lines.fa', 'w')
    for j in g_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('c_lines.fa', 'w')
    for j in c_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('f_lines.fa', 'w')
    for j in f_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('o_lines.fa', 'w')
    for j in o_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('h_lines.fa', 'w')
    for j in h_lines:
        to_write.write(j)
    to_write.close()

    to_write = open('p_lines.fa', 'w')
    for j in p_lines:
        to_write.write(j)
    to_write.close()

def step2():
    
    id_to_label = {}
    
    with open("cog_to_label.json", "r") as f:
        cog_to_label = json.load(f)
    
    #5951
    for i in range(1, 100):
        num = str(i).zfill(4)
        filename = "fasta/COG" + num + ".tsv.gz"
        #print(filename)
        if os.path.exists(filename):
            with gzip.open(filename, "rt") as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    id = row[0]
                    label = cog_to_label.get("COG" + num)
                    if label is not None:
                        #print("COG" + num + ": ", id, label)
                        id_to_label[id] = label
                    #else:
                    #    print("Label not found.")
        #else:
        #    print("File does not exist.")
            
    with open("protein_id_to_label.json", "w") as f:
        json.dump(id_to_label, f)
        #print("ID TO LABEL TEST")
        #print(id_to_label["ABX12048.1"])

def step1():
    with open("cog-20.def.tab", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        cog_to_label = {row[0]: row[1][0] for row in reader}
    
    with open("cog_to_label.json", "w") as f:
        json.dump(cog_to_label, f)
        
step1()
step2()
main()