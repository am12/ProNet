# check if a file with the name COG0001.tsv.gz exists in the current directory. If it does, print "File exists." Otherwise, print "File does not exist."

import os
    
# iterate from 1 to 5951
for i in range(1, 5951):
    num = str(i).zfill(4)
    url = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/fasta/COG" + num + ".tsv.gz"
    # response = requests.get(url)
    if os.path.exists("COG" + num + ".tsv.gz"):
        # print("File exists.")
        pass
    else:
        # write file number does not exist to a file
        with open("file_does_not_exist.txt", "a") as f:
            f.write(num + "\n")