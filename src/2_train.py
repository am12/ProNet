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

def split_data(datafile):

    
    return train_datafile, test_datafile


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


def train(train_datafile, train_log_file, batch_size, lr=0.01, device_str=None):
    
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
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f'\t[Info] Done initializing model.', flush=True)

    # load data to train the model on 
    print(f'\t[Info] Loading data...', flush=True)
    train_loader = get_dataloader(batch_size, 'train', train_datafile, False, seed=42)
    print(f'\t[Info] Done loading data.', flush=True)

    # time after iteration
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # TRAINING LOOP
    print('Starting training.')
    start_time = time.time()
    model.train()
    junc_counter = 0
    pbar = Bar(f'Training ProNet... ', max=len(train_loader))
    with torch.no_grad(), open(train_log_file, 'w') as fw:
        for batch_idx, data in enumerate(train_loader):

            seqs, labels, prot_id, cog_id = data
            seqs = seqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)

            
            seqs = torch.permute(seqs, (0, 2, 1))
            labels = torch.permute(labels, (0, 2, 1))
            loss, yp = model_fn(seqs, labels, model, criterion)
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

            # increment the progress bar
            pbar.next()

    pbar.finish()
    fw_junc_scores.close()
    return out_score_f


if __name__ == '__main__':


    train_dataloader = get_dataloader(BATCH_SIZE, 'train', )
    train_dataset = HandsDataset("train.csv", None, CUDA)  # Adjust path and normalization as necessary
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = HandsDataset("test.csv", None, CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    train_datafile, test_datafile = split_data()
    train_log_file = '../results/2/train.log'
    os.makedirs(os.path.dirname(train_log_file), exist_ok=True)
    train(train_datafile, train_log_file, BATCH_SIZE)