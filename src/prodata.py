# load data into a dataloader
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# pad with X and one-hot encode 
# later will try to experiment with other kinds of embeddings 

def get_dataloader(batch_size, mode, dataset_file, shuffle, seed=None, segment_len=None, verbose=True):
    dataset = ProData(mode, dataset_file, shuffle, seed, segment_len, verbose)
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False, # because the ProData class handles shuffling already
        drop_last = False,
        pin_memory = True,
    )
    return loader

def create_datapoints(seq, max_len, label):
    '''Truncates, pads, and performs one-hot encoding of the protein sequence and labels'''
    
    # uppercase and truncate or pad with 'X' so all are equal length 
    seq = seq.upper()[:max_len] + 'X' * (max_len - len(seq))
    
    # mapping from amino acids to indices
    aa_to_index = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }

    # convert sequence to indices, setting X to 0
    indexed_seq = [aa_to_index.get(aa, 0) for aa in seq]

    # convert sequence and labels into numpy arrays
    X0 = np.array(indexed_seq, dtype=np.int32)

    # one-hot encode the input sequence
    X = np.zeros((len(X0), 20))  # create an array of zeros for 20 amino acids
    for i, index in enumerate(X0):
        if index > 0:
            X[i, index-1] = 1  # set the appropriate index to 1, shifting by 1 because index 0 is for 'X' as all zeros

    # label mapping from letter to index
    label_to_index = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,
        'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Z': 24
    }

    # one-hot encode the single letter label
    Y = np.zeros((25,))  # create a zero vector of length 25
    Y[label_to_index[label.upper()]] = 1  # set the index corresponding to the label

    return X, Y


class ProData(Dataset):
    def __init__(self, mode, dataset_file, shuffle, seed=None, segment_len=None, verbose=True):
        '''
        Parameters: 
            - mode (str) -> 'train' or 'test', just a label no function
            - dataset_file (str) -> path to the created dataset 
            - shuffle (bool) -> whether to shuffle the data (generally yes for train, no for test)
            - seed (int) -> random seed to use if given (default: None)
            - segment_len (int) -> length to truncate protein sequence to (default: None -> will not truncate)
            - verbose (bool) -> whether to print out status (default: True)
        '''
        self.mode = mode
        self.segment_len = segment_len # will truncate to this length 
        self.data = []
        self.indices = [] # shuffle indices
        
        if mode not in ['train', 'test']:
            raise ValueError('mode must be either "train" or "test".')
            
        # set the random state
        if seed:
            random.seed(seed)
        
        # parse the dataset
        if verbose:
            print(f'\t[INFO] Creating {mode} dataset from source: {dataset_file}')
        with open(dataset_file, 'r') as f:

            lines = f.read().splitlines()
            data = [line.split(',') for line in lines]
            # if no truncation, find the maximum length of all sequences 
            if segment_len == None: 
                # max_len = max([len(row[3]) for row in data])
                max_len = 600 # based on domain knowledge
            # otherwise, truncate to set length
            else:
                max_len = segment_len
            
            if verbose:
                lens = sorted([len(row[3]) for row in data])
                print(f'\t[STAT] Maximum protein length: {max(lens)}')
                print(f'\t[STAT] Minimum protein length: {min(lens)}')
                print(f'\t[STAT] Mean protein length: {sum(lens) / len(lens):.6f}')
                print(f'\t[STAT] Median protein length: {lens[len(lens)//2]}')
                print(f'\t[STAT] Will truncate proteins to length {max_len}.')

            pidx = 0
            for row in data:
                seq, label = row[3], row[2]
                prot_id, cog_id = row[0], row[1]
                
                # one-hot encoding and padding
                X, Y = create_datapoints(seq, max_len, label)
                X = torch.Tensor(np.array(X))
                Y = torch.Tensor(np.array(Y))

                # add to dataset 
                self.data.append([X, Y, prot_id, cog_id])
                
                # reporting 
                if verbose and (pidx + 1) % 10000 == 0:
                    print(f'\t[INFO] {pidx + 1} proteins loaded.')

                pidx += 1

            if verbose:
                print(f'\t[STAT] X.shape: {X.shape} | Y.shape: {Y.shape}')

        # handle indices
        indices = list(range(len(self.data)))
        if shuffle:
            random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.indices = indices

        if verbose: 
            print(f'\t[INFO] {pidx} proteins loaded total.')

    def get_mode(self):
        return self.mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index][0]
        label = self.data[index][1]
        prot_id = self.data[index][2]
        cog_id = self.data[index][3]
        sequence = torch.flatten(sequence, start_dim=1)
        return sequence, label, prot_id, cog_id