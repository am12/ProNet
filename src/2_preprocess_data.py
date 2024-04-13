# load data into a dataloader
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# pad with X and one-hot encode 
# later will try to experiment with other kinds of embeddings 

class ProData(Dataset):
    def __init__(self, mode, dataset_file, shuffle, seed=None, segment_len=None, verbose=True):
        '''
        Parameters: 
            - mode (str) -> 'train' or 'test'
            - dataset_file (str) -> path to the created dataset 
            - shuffle (bool) -> whether to shuffle the data (generally yes for train, no for test)
            - seed (int) -> random seed to use if given (default: None)
            - segment_len (int) -> length to truncate protein sequence to (default: None -> will not truncate)
            - verbose (bool) -> whether to print out status (default: True)
        '''
        self.segment_len = segment_len # will truncate to this length 
        self.data = []
        self.indices = [] # shuffle indices
        self.mode = mode
        self.shuffle = shuffle
        
        # set the random state
        if seed:
            random.seed(seed)
        
        # parse the dataset
        with open(dataset_file, 'r') as f:

            lines = f.read().splitlines()
            data = np.array([line.split(',') for line in lines])
            
            # if no truncation, find the maximum length of all sequences 
            if segment_len == None: 
                max_len = np.max(data[:, 3])
            # otherwise, truncate to set length
            else:
                max_len = segment_len
            if verbose:
                print(f'\t[INFO] Maximum protein length: {np.max(data[:, 3])}')
                print(f'\t[INFO] Minimum protein length: {np.min(data[:, 3])}')
                print(f'\t[INFO] Mean protein length: {np.mean(data[:, 3])}')
                print(f'\t[INFO] Median protein length: {np.median(data[:, 3])}')
                print(f'\t[INFO] Will truncate proteins to length {max_len}.')

            pidx = 0
            for row in data:
                seq, label = row[3], row[2]
                prot_id, cog_id = row[0], row[1]
                
                # one-hot encoding and padding
                X, Y = create_datapoints(seq, max_len)
                X = torch.Tensor(np.array(X))
                Y = torch.Tensor(np.array(Y)[0])

                # add to dataset 
                self.data.append([X, Y, prot_id, cog_id])
                
                # reporting 
                if verbose and pidx % 100000 == 0:
                    print(f'\t[INFO] {pidx} proteins loaded.')

                pidx += 1

        # handle indices
        indices = list(range(len(self.data)))
        if shuffle:
            random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.indices = indices

        if verbose: 
            print(f'\t[INFO] {pidx} junctions loaded.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index][0]
        label = self.data[index][1]
        prot_id = self.data[index][2]
        cog_id = self.data[index][3]
        sequence = torch.flatten(sequence, start_dim=1)
        return sequence, label, prot_id, cog_id

def create_datapoints(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')
    pattern = r'[^1234]'
    # Replace non-ACGT characters with 0
    seq = re.sub(pattern, '0', seq)
    jn_start = JUNC_START
    jn_end = JUNC_END

    #######################################
    # predicting pb for every bp
    #######################################
    X0 = np.asarray(list(map(int, list(seq))))
    Y0 = [np.zeros(SEQ_LEN) for t in range(1)]
    if strand == '+':
        for t in range(1):
            Y0[t][jn_start] = 2
            Y0[t][jn_end] = 1
    X, Y = one_hot_encode(X0, Y0)
    return X, Y


def get_dataloader(batch_size, n_workers, output_file, shuffle, repeat_idx):
    testset = myDataset('test', output_file, shuffle, SEQ_LEN)
    test_loader = DataLoader(
        testset,
        batch_size = batch_size,
        drop_last = False,
        pin_memory = True,
    )
    return test_loader