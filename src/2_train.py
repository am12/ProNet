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
import math
import random
from tqdm import tqdm 

### GLOBAL VARIABLES ### 
NUM_EPOCHS = 10
BATCH_SIZE = 100
RANDOM_SEED = 42 # this seed is for shuffling data
SCHEDULER_STEPS = 1000
experiment_number = 2
input_dir = '../results/1/'
out_dir = '../results/2/'
os.makedirs(os.path.dirname(out_dir)+'/runs/experiment_'+str(experiment_number)+'/models/', exist_ok=True)

def same_seeds(seed):
    '''Fix random seeds for PyTorch'''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0) # this seed is for torch 

# def categorical_crossentropy_2d(y_pred, y_true, seq_weight=5, gamma=2):
#     '''Computes weighted categorical CE loss for 2D tensor, appling focal loss modification for hard examples.'''
#     return - torch.mean(y_true[:, 0, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 0, :]), gamma ), torch.log(y_pred[:, 0, :]+1e-10) )
#                         + seq_weight * y_true[:, 1, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 1, :]), gamma ), torch.log(y_pred[:, 1, :]+1e-10) )
#                         + seq_weight * y_true[:, 2, :] * torch.mul( torch.pow( torch.sub(1, y_pred[:, 2, :]), gamma ), torch.log(y_pred[:, 2, :]+1e-10) ))

def get_cosine_schedule_with_warmup(
      optimizer: Optimizer,
      num_warmup_steps: int,
      num_training_steps: int,
      num_cycles: float = 0.5,
      last_epoch: int = -1,
    ):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

    Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def split_data(datafile, test_ratio=0.15, indices=None, seed=None, save_loader=False):
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

    if save_loader:
        print(f'\t[INFO] Saving dataloaders to PyTorch files...')
        train_loader_path = f'{base_dir}/train_dataloader.pt'
        test_loader_path = f'{base_dir}/test_dataloader.pt'
        torch.save(train_loader, train_loader_path)
        torch.save(test_loader, test_loader_path)
        print(f'\t[INFO] Saved.')

        return train_loader_path, test_loader_path

    return train_file_path, test_file_path


def train_single_epoch(epoch_num, model, device, train_loader, logger, criterion, optimizer, scheduler):
    
    model.train()
    print(f'\t[INFO] Model in train mode.')

    pbar = tqdm(total=len(train_loader), ncols=0, desc="Train", unit=" step")
    for batch_idx, data in enumerate(train_loader): 
            
        ### PREDICTION ###
        seqs, labels, prot_id, cog_id = data
        seqs = seqs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        seqs = torch.permute(seqs, (0, 2, 1))
        
        # forward pass
        pred = model(seqs)
        loss = criterion(pred, labels)

        # backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        ### LOGGING ###
        iter_num = epoch_num * len(train_loader) + batch_idx
        logger.add_scalar('Loss/train', loss.item(), iter_num)

        if (batch_idx + 1) % 10 == 0:
            # add histograms of model parameters to inspect their distributions
            for name, weight in model.named_parameters():
                logger.add_histogram(name, weight, iter_num)
                logger.add_histogram(f'{name}.grad', weight.grad, iter_num)
        
        pbar.update(1)
        pbar.set_postfix(
            batch_idx=batch_idx,
            train_idx=epoch_num * len(train_loader),
            ce_loss=f"{loss:.3f}",
            lr=f'{scheduler.get_last_lr()[0]:.6f}',
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

def evaluate_single_epoch(epoch_num, model, device, test_loader, stats_file, criterion):
    
    model.eval()
    print(f'\t[INFO] Model in evaluation mode.')

    total = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), ncols=0, desc="Test", unit=" step")
        for batch_idx, data in enumerate(test_loader):

            ### PREDICTION ###
            seqs, labels, prot_id, cog_id = data
            seqs = seqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            seqs = torch.permute(seqs, (0, 2, 1))
            
            pred = model(seqs)
            loss = criterion(pred, labels)

            ### LOGGING ###
            # print(pred.shape, labels.shape)
            pred_lab = torch.argmax(pred,dim=1)
            labels_lab = torch.argmax(labels,dim=1)
            # print(pred_lab.shape, pred_lab, labels_lab.shape, labels_lab)
            total += labels_lab.size(0)
            correct += (pred_lab == labels_lab).sum().item() 
                
            pbar.update(1)
            pbar.set_postfix(
                batch_id=batch_idx,
                test_idx=epoch_num * len(test_loader),
                ce_loss=f"{loss:.3f}",
                accuracy=f'{100*correct/total:.3f}%'
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

    print(f'Accuracy of the model on the test dataset: {100 * correct / total:.4f}%')
    # print(f'Epoch {epoch_idx+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Donor top-k Acc: {epoch_donor_acc/len(train_loader):.3f} | Acceptor top-k Acc: {epoch_acceptor_acc/len(train_loader):.3f}')
    # print(f'Junction Precision: {J_G_TP/(J_G_TP+J_G_FP):.5f} | Junction Recall: {J_G_TP/(J_G_TP+J_G_FN):.5f} | TP: {J_G_TP} | FN: {J_G_FN} | FP: {J_G_FP} | TN: {J_G_TN}')
    # print(f'Donor Precision   : {D_G_TP/(D_G_TP+D_G_FP):.5f} | Donor Recall   : {D_G_TP/(D_G_TP+D_G_FN):.5f} | TP: {D_G_TP} | FN: {D_G_FN} | FP: {D_G_FP} | TN: {D_G_TN}')
    # print(f'Acceptor Precision: {A_G_TP/(A_G_TP+A_G_FP):.5f} | Acceptor Recall: {A_G_TP/(A_G_TP+A_G_FN):.5f} | TP: {A_G_TP} | FN: {A_G_FN} | FP: {A_G_FP} | TN: {A_G_TN}')
    # print ("Learning rate: %.5f" % (get_lr(optimizer)))
    # print("\n\n")

def train(train_datafile, test_datafile, train_statsfile, batch_size, lr=1e-3, num_epochs=16, device_str=None, saved_loader=False):
    '''
    The training runner function. 

    train_datafile (str) -> path to train data csv file. OR path to train dataloader pt file if saved_loader == True
    test_datafile (str) -> path to test data csv file. OR path to test dataloader pt file if saved_loader == True
    train_statsfile (str) -> path to evaluation statistics log file. 
    batch_size (int) -> size of dataloader batch
    lr (float) -> learning rate
    num_epochs (int) -> number of epochs to train
    device_str (str) -> device name
    saved_loader (bool) -> whether to directly load the dataloader
    '''

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
    print(f'\t[SYS] Running model in {device_str} mode.', flush=True)

    # load data to train the model on 
    print(f'\t[SYS] Loading data...', flush=True)
    if saved_loader: 
        train_loader = torch.load(train_datafile)
        test_loader = torch.load(test_datafile)
    else:
        train_loader = get_dataloader(batch_size, 'train', train_datafile, True, seed=RANDOM_SEED)
        test_loader = get_dataloader(batch_size, 'test', test_datafile, False, seed=RANDOM_SEED)  
    print(f'\t[SYS] Done loading data.', flush=True)

    # load a new instance of ProNet to train
    print(f'\t[SYS] Initializing new ProNet model...', flush=True)
    model = ProNet().to(device)
    criterion = torch.nn.CrossEntropyLoss() # maybe try focal loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f'\t[SYS] Done initializing model.', flush=True)

    # get scheduler
    print(f'\t[SYS] Getting schedule...', flush=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, SCHEDULER_STEPS, len(train_loader)*NUM_EPOCHS)
    print(f"\t[SYS] Initialized scheduler. Warmup steps: {SCHEDULER_STEPS} | Total steps: {len(train_loader)*NUM_EPOCHS}", flush=True)

    ## Look into double dipping

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
            print(f'Epoch {epoch_num+1}/{num_epochs}:')
            train_single_epoch(epoch_num, model, device, train_loader, logger, criterion, optimizer, scheduler)
            evaluate_single_epoch(epoch_num, model, device, test_loader, stats_file, criterion)
            torch.save(model, f'{out_dir}runs/experiment_{experiment_number}/models/pronet_epoch-{epoch_num+1}.pt')
        
    print("--- %s minutes ---" % ((time.time() - start_time)/60.0))


if __name__ == '__main__':
    use_saved_loader = True
    # train_data, test_data = split_data(f'{input_dir}dataset.csv', save_loader=use_saved_loader)
    train_data = '../results/1/train_dataloader.pt'
    test_data = '../results/1/test_dataloader.pt'
    train_statsfile = f'{out_dir}train_statistics.txt'
    train(train_data, test_data, train_statsfile, BATCH_SIZE, saved_loader=use_saved_loader)