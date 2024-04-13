# load data into a dataloader
import random

# pad with X and one-hot encode 

class ProData(Dataset):
    def __init__(self, type, of, shuffle, segment_len=800):
        self.segment_len = segment_len
        self.data = []
        self.indices = []
        pidx = 0
        with open(of, 'r') as f:
            lines = f.read().splitlines()
            seq_name = ''
            seq = ''
            for line in lines:
                if pidx % 2 == 0:
                    seq_name = split_seq_name(line)
                elif pidx % 2 == 1:
                    seq = line
                    if seq[0] == '>':
                        seq_name = line
                        continue
                    X, Y = create_datapoints(seq, '+')
                    X = torch.Tensor(np.array(X))
                    Y = torch.Tensor(np.array(Y)[0])
                    if X.size()[0] != 800:
                        print('seq_name: ', seq_name)
                        print(X.size())
                        print(Y.size())
                    self.data.append([X, Y, seq_name])
                pidx += 1
                if pidx % 100000 == 0:
                    print('\t', pidx//2, ' junctions loaded.')

        index_shuf = list(range(len(self.data)))
        if shuffle:
            random.shuffle(index_shuf)
        list_shuf = [self.data[i] for i in index_shuf]
        self.data = list_shuf
        self.indices = index_shuf
        print('\t', pidx//2, ' junctions loaded.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.data[index][0]
        label = self.data[index][1]
        seq_name = self.data[index][2]
        feature = torch.flatten(feature, start_dim=1)
        return feature, label, seq_name

def get_dataloader(batch_size, n_workers, output_file, shuffle, repeat_idx):
    testset = myDataset('test', output_file, shuffle, SEQ_LEN)
    test_loader = DataLoader(
        testset,
        batch_size = batch_size,
        drop_last = False,
        pin_memory = True,
    )
    return test_loader