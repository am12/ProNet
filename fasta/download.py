import requests

for i in range(1, 5951):
    num = str(i).zfill(4)
    url = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/fasta/COG" + num + ".tsv.gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open('COG' + num + ".tsv.gz", 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")
    
    