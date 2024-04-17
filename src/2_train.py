# run training loop
import os
import torch 
from torch.nn import Module, BatchNorm1d, LeakyReLU, Conv1d, ModuleList, Softmax, Sigmoid, Flatten, Dropout2d, Linear
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pronet import ProNet
from prodata import ProData, get_dataloader
import time
import random
from tqdm import tqdm 

### GLOBAL VARIABLES ### 
NUM_EPOCHS = 10
BATCH_SIZE = 100
RANDOM_SEED = 42 # this seed is for shuffling data
experiment_number = 1
input_dir = '../results/1/'
out_dir = '../results/2/'
os.makedirs(os.path.dirname(out_dir), exist_ok=True)


def same_seeds(seed):
    '''Fix random seeds for PyTorch'''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0) # this seed is for torch 

def categorical_crossentropy_2d(y_pred, y_true, seq_weight=5, gamma=2):
    '''Computes weighted categorical CE loss for 2D tensor, appling focal loss modification for hard examples.'''
    return - torch.mean(y_true[:, 0, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 0, :]), gamma ), torch.log(y_pred[:, 0, :]+1e-10) )
                        + seq_weight * y_true[:, 1, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 1, :]), gamma ), torch.log(y_pred[:, 1, :]+1e-10) )
                        + seq_weight * y_true[:, 2, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 2, :]), gamma ), torch.log(y_pred[:, 2, :]+1e-10) ))

def split_data(datafile, test_ratio=0.15, indices=None, seed=None):
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
    base_dir = os.path.dirname(datafile) + '/'
    train_file_path = f'{base_dir}train_data.csv'
    test_file_path = f'{base_dir}test_data.csv'
    with open(train_file_path, 'w') as file:
        file.writelines(train_lines)
    with open(test_file_path, 'w') as file:
        file.writelines(test_lines)
    print(f'\t[INFO] Written datasets to: {base_dir}.')

    return train_file_path, test_file_path

def train_single_epoch(epoch_num, model, device, train_loader, logger, optimizer, criterion=None):
    
    model.train()
    print(f'\t[INFO] Model in train mode.')

    running_loss = 0.0
    pbar = tqdm(total=len(train_loader), ncols=0, desc="Train", unit=" step")
    for batch_idx, data in enumerate(train_loader): 
            
        ### PREDICTION ###
        seqs, labels, prot_id, cog_id = data
        seqs = seqs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        seqs = torch.permute(seqs, (0, 2, 1))
        print('\n', seqs.shape, labels.shape)
        
        # forward pass
        pred = model(seqs)
        if criterion:
            loss = criterion(pred, labels)
        else:
            loss = categorical_crossentropy_2d(pred, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### LOGGING ###
        iter_num = epoch_num * len(train_loader) + batch_idx
        logger.add_scalar('Loss/train', loss.item(), iter_num)

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f'\t[INFO] Epoch {epoch_num + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 10:.6f}')     
            running_loss = 0.0
        
            # add histograms of model parameters to inspect their distributions
            for name, weight in model.named_parameters():
                logger.add_histogram(name, weight, iter_num)
                logger.add_histogram(f'{name}.grad', weight.grad, iter_num)
        
        pbar.update(1)
        pbar.set_postfix(
            batch_id=batch_idx,
            # idx_train=len(train_loader)*BATCH_SIZE,
            # loss=f"{batch_loss:.6f}",
            # A_auprc = f"{A_auprc:.6f}",
            # D_auprc = f"{D_auprc:.6f}",
            # A_Precision=f"{A_TP/(A_TP+A_FP+1e-6):.6f}",
            # A_Recall=f"{A_TP/(A_TP+A_FN+1e-6):.6f}",
            # D_Precision=f"{D_TP/(D_TP+D_FP+1e-6):.6f}",
            # D_Recall=f"{D_TP/(D_TP+D_FN+1e-6):.6f}",
            # J_Precision=f"{J_TP/(J_TP+J_FP+1e-6):.6f}",
            # J_Recall=f"{J_TP/(J_TP+J_FN+1e-6):.6f}"
        )  

    pbar.close()

    # print(f'Accuracy of the model on the train dataset: {100 * correct / total}%', file=stats_file)
    # print(f'Epoch {epoch_idx+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Donor top-k Acc: {epoch_donor_acc/len(train_loader):.3f} | Acceptor top-k Acc: {epoch_acceptor_acc/len(train_loader):.3f}')
    # print(f'Junction Precision: {J_G_TP/(J_G_TP+J_G_FP):.5f} | Junction Recall: {J_G_TP/(J_G_TP+J_G_FN):.5f} | TP: {J_G_TP} | FN: {J_G_FN} | FP: {J_G_FP} | TN: {J_G_TN}')
    # print(f'Donor Precision   : {D_G_TP/(D_G_TP+D_G_FP):.5f} | Donor Recall   : {D_G_TP/(D_G_TP+D_G_FN):.5f} | TP: {D_G_TP} | FN: {D_G_FN} | FP: {D_G_FP} | TN: {D_G_TN}')
    # print(f'Acceptor Precision: {A_G_TP/(A_G_TP+A_G_FP):.5f} | Acceptor Recall: {A_G_TP/(A_G_TP+A_G_FN):.5f} | TP: {A_G_TP} | FN: {A_G_FN} | FP: {A_G_FP} | TN: {A_G_TN}')
    # print ("Learning rate: %.5f" % (get_lr(optimizer)))
    # print("\n\n")

def evaluate_single_epoch(epoch_num, model, device, test_loader, stats_file, criterion=None):
    
    model.eval()
    print(f'\t[INFO] Model in evaluation mode.')

    total = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), ncols=0, desc="Test", unit=" step")
        for batch_idx, data in enumerate(test_loader):
            seqs, labels, prot_id, cog_id = data
            seqs = seqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            seqs = torch.permute(seqs, (0, 2, 1))
            labels = torch.permute(labels, (0, 2, 1))
            
            pred = model(seqs)
            if criterion:
                loss = criterion(pred, labels)
            else:
                loss = categorical_crossentropy_2d(pred, labels)

            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.update(1)
            pbar.set_postfix(
                batch_id=batch_idx,
                # idx_test=len(test_loader)*BATCH_SIZE,
                # loss=f"{batch_loss:.6f}",
                # A_auprc = f"{A_auprc:.6f}",
                # D_auprc = f"{D_auprc:.6f}",
                # A_Precision=f"{A_TP/(A_TP+A_FP+1e-6):.6f}",
                # A_Recall=f"{A_TP/(A_TP+A_FN+1e-6):.6f}",
                # D_Precision=f"{D_TP/(D_TP+D_FP+1e-6):.6f}",
                # D_Recall=f"{D_TP/(D_TP+D_FN+1e-6):.6f}",
                # J_Precision=f"{J_TP/(J_TP+J_FP+1e-6):.6f}",
                # J_Recall=f"{J_TP/(J_TP+J_FN+1e-6):.6f}"
            )
    
        pbar.close()

    print(f'Accuracy of the model on the test dataset: {100 * correct / total}%', file=stats_file)
    # print(f'Epoch {epoch_idx+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Donor top-k Acc: {epoch_donor_acc/len(train_loader):.3f} | Acceptor top-k Acc: {epoch_acceptor_acc/len(train_loader):.3f}')
    # print(f'Junction Precision: {J_G_TP/(J_G_TP+J_G_FP):.5f} | Junction Recall: {J_G_TP/(J_G_TP+J_G_FN):.5f} | TP: {J_G_TP} | FN: {J_G_FN} | FP: {J_G_FP} | TN: {J_G_TN}')
    # print(f'Donor Precision   : {D_G_TP/(D_G_TP+D_G_FP):.5f} | Donor Recall   : {D_G_TP/(D_G_TP+D_G_FN):.5f} | TP: {D_G_TP} | FN: {D_G_FN} | FP: {D_G_FP} | TN: {D_G_TN}')
    # print(f'Acceptor Precision: {A_G_TP/(A_G_TP+A_G_FP):.5f} | Acceptor Recall: {A_G_TP/(A_G_TP+A_G_FN):.5f} | TP: {A_G_TP} | FN: {A_G_FN} | FP: {A_G_FP} | TN: {A_G_TN}')
    # print ("Learning rate: %.5f" % (get_lr(optimizer)))
    # print("\n\n")

def train(train_datafile, test_datafile, train_statsfile, batch_size, lr=1e-3, num_epochs=16, device_str=None):
    
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f'\t[Info] Done initializing model.', flush=True)

    # load data to train the model on 
    print(f'\t[Info] Loading data...', flush=True)
    # train_loader = get_dataloader(batch_size, 'train', train_datafile, True, seed=RANDOM_SEED)
    # test_loader = get_dataloader(batch_size, 'test', test_datafile, False, seed=RANDOM_SEED)
    ### to expedite
    # torch.save(train_loader, '../results/1/train_dataloader.pt')
    # torch.save(test_loader, '../results/1/test_dataloader.pt')
    train_loader = torch.load('../results/1/train_dataloader.pt')
    test_loader = torch.load('../results/1/test_dataloader.pt')
    print(f'\t[Info] Done loading data.', flush=True)

    # time after iteration
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #####################
    ### TRAINING LOOP ###
    #####################

    # initialize the SummaryWriter
    logger = SummaryWriter(f'{out_dir}runs/experiment_{experiment_number}')

    print('Starting training...')
    start_time = time.time()
    with open(train_statsfile, 'w') as stats_file:
        for epoch_num in range(num_epochs):
            print(f'Epoch {epoch_num}/{num_epochs}:')
            train_single_epoch(epoch_num, model, device, train_loader, logger, optimizer, criterion)
            evaluate_single_epoch(epoch_num, model, device, test_loader, stats_file)
            torch.save(model, f'{out_dir}runs/experiment_{experiment_number}/models/pronet_epoch-{epoch_num}.pt')
        
    print("--- %s minutes ---" % ((time.time() - start_time)/60.0))


if __name__ == '__main__':

    # train_datafile, test_datafile = split_data(f'{input_dir}dataset.csv')
    ### TO EXPEDITE 
    train_datafile = '../results/1/train_data.csv'
    test_datafile = '../results/1/test_data.csv'
    train_statsfile = f'{out_dir}train_statistics.txt'
    train(train_datafile, test_datafile, train_statsfile, BATCH_SIZE)