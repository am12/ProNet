# run training loop
import torch 
from torch.nn import Module, BatchNorm1d, LeakyReLU, Conv1d, ModuleList, Softmax, Sigmoid, Flatten, Dropout2d, Linear
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from pronet import ProNet
from prodata import ProData
from preprocess_data import ProData, get_dataloader
import time

BATCH_SIZE = 16
RANDOM_SEED = 42
experiment_number = 1
out_dir = '../results/2/'
os.makedirs(os.path.dirname(out_dir), exist_ok=True)

def split_data(datafile, test_ratio=0.2, indices=None, seed=None):
    '''Create a training and testing split of the datafiles'''

    # read all lines from the input file
    with open(datafile, 'r') as file:
        lines = file.readlines()
    
    # shuffle the lines based on predefined method (either given indices or random shuffle)
    if indices: 
        assert len(indices) == len(lines) # sanity check 
        lines = [lines[idx] for idx in indices]
    else:
        if seed:
            random.seed(seed)
        random.shuffle(lines)

    # calculate split index
    split_idx = int(len(lines) * (1 - test_ratio))
    
    # split the data
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    print(f'\t[INFO] Split data. Training size: {len(train_lines)} | Testing size: {len(test_lines)}')
    
    # write data to respective files
    base_dir = os.path.dirname(datafile)
    train_file_path = f'{base_dir}train_data.csv'
    test_file_path = f'{base_dir}test_data.csv'
    with open(train_file_path, 'w') as file:
        file.writelines(train_lines)
    with open(test_file_path, 'w') as file:
        file.writelines(test_lines)
    print(f'\t[INFO] Written datasets to: {base_dir}.')

    return train_file_path, test_file_path


def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0


def evaluate(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test dataset: {100 * correct / total}%')


def categorical_crossentropy_2d(y_pred, y_true, seq_weight=5, gamma=2):
    return - torch.mean(y_true[:, 0, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 0, :]), gamma ), torch.log(y_pred[:, 0, :]+1e-10) )
                        + seq_weight * y_true[:, 1, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 1, :]), gamma ), torch.log(y_pred[:, 1, :]+1e-10) )
                        + seq_weight * y_true[:, 2, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 2, :]), gamma ), torch.log(y_pred[:, 2, :]+1e-10) ))


def train(train_datafile, test_datafile, train_logfile, batch_size, lr=1e-3, num_epochs=10, device_str=None):
    
    print('Setting up training variables...')
    start_time = time.time()

    # find the right device to use
    if not device_str:
        device_str = 'cpu'
        if torch.cuda.is_available(): 
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
    device = torch.device(device_str)
    print(f'\t[Info] Running model in {device_str} mode.', flush=True)

    # load a new instance of ProNet to train
    print(f'\t[Info] Initializing new ProNet model...', flush=True)
    model = ProNet().to(device)
    #criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f'\t[Info] Done initializing model.', flush=True)

    # load data to train the model on 
    print(f'\t[Info] Loading data...', flush=True)
    train_loader = get_dataloader(batch_size, 'train', train_datafile, False, seed=RANDOM_SEED)
    print(f'\t[Info] Done loading data.', flush=True)

    # time after iteration
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #####################
    ### TRAINING LOOP ###
    #####################

    # initialize the SummaryWriter
    writer = SummaryWriter(f'{out_dir}runs/experiment_{experiment_number}')

    print('Starting training.')
    start_time = time.time()
    log = open(train_logfile, 'w')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader): 
                
            ### PREDICTION ###
            seqs, labels, prot_id, cog_id = data
            seqs = seqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            seqs = torch.permute(seqs, (0, 2, 1))
            labels = torch.permute(labels, (0, 2, 1))
            
            # forward pass
            outputs = model(seqs)
            #loss = criterion(outputs, labels)
            loss = categorical_crossentropy_2d(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ### LOGGING ###
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

            running_loss += loss.item()
            if batch_idx % 10 == 9:
                print(f'\t[INFO] Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.6f}')     
                running_loss = 0.0
            
            # add histograms of model parameters to inspect their distributions
            for name, weight in model.named_parameters():
                writer.add_histogram(name, weight, epoch)
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)
            
            is_expr = (labels.sum(axis=(1,2)) >= 1)
            A_YL = labels[is_expr, 1, :].to('cpu').detach().numpy()
            A_YP = yp[is_expr, 1, :].to('cpu').detach().numpy()
            D_YL = labels[is_expr, 2, :].to('cpu').detach().numpy()
            D_YP = yp[is_expr, 2, :].to('cpu').detach().numpy()

            donor_labels, donor_scores, acceptor_labels, acceptor_scores = get_donor_acceptor_scores(D_YL, A_YL, D_YP, A_YP)
            
            # for idx in range(len(yp)):
                
                # eles = seqname[idx].split(';')
                # if len(eles) == 7:
                #     chr, start, end, strand, name, aln_num, trans = eles
                #     if strand == '+':
                #         fw_junc_scores.write(f'{chr}\t{str(start)}\t{str(end)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\t{trans}\n')
                #     elif strand == '-':
                #         fw_junc_scores.write(f'{chr}\t{str(end)}\t{str(start)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\t{trans}\n')

                # else:
                #     chr, start, end, strand, name, aln_num = eles
                #     if strand == '+':
                #         fw_junc_scores.write(f'{chr}\t{str(start)}\t{str(end)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\n')
                #     elif strand == '-':
                #         fw_junc_scores.write(f'{chr}\t{str(end)}\t{str(start)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\n')
                
                # junc_counter += 1 

        
     

    print("--- %s seconds ---" % (time.time() - start_time))


    log.close()

    return out_score_f

if __name__ == '__main__':

    train_dataloader = get_dataloader(BATCH_SIZE, 'train', )
    train_dataset = HandsDataset("train.csv", None, CUDA)  # Adjust path and normalization as necessary
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = HandsDataset("test.csv", None, CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    

    # training 
    train_datafile, test_datafile = split_data()
    train_logfile = '../results/2/train.log'
    train(train_datafile, test_datafile, train_logfile, BATCH_SIZE)