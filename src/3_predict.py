# run prediction
import torch
from torch.nn import Module, BatchNorm1d, LeakyReLU, Conv1d, ModuleList, Softmax, Sigmoid, Flatten, Dropout2d, Linear
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader




def splam_prediction(junction_fasta, out_score_f, model_path, batch_size, device_str):
    BATCH_SIZE = int(batch_size)
    N_WORKERS = None
    if device_str == "NONE":
        device_str = 'cpu'
        if torch.cuda.is_available(): 
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
    device = torch.device(device_str)

    print(f'[Info] Running model in "'+ device_str+'" mode')
    print(f'[Info] Loading model ... (' + model_path + ')', flush = True)
    # model = torch.jit.load(model_path)
    print("model = torch.load(model_path)!!")
    model = torch.load(model_path)
    model = model.to(device)

    print(f'[Info] Done loading model', flush = True)
    print(f'[Info] Loading data ...', flush = True)
    test_loader = get_dataloader(BATCH_SIZE, N_WORKERS, junction_fasta, False, str(0))
    print(f'[Info] Done loading data', flush = True)
    
    criterion = torch.nn.BCELoss()
    fw_junc_scores = open(out_score_f, 'w')

    model.eval()
    junc_counter = 0
    pbar = Bar('[Info] SPLAM! ', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # DNAs:  torch.Size([40, 800, 4])
            # labels:  torch.Size([40, 1, 800, 3])
            DNAs, labels, seqname = data
            DNAs = DNAs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)

            DNAs = torch.permute(DNAs, (0, 2, 1))
            labels = torch.permute(labels, (0, 2, 1))
            loss, yp = model_fn(DNAs, labels, model, criterion)
            is_expr = (labels.sum(axis=(1,2)) >= 1)
            A_YL = labels[is_expr, 1, :].to('cpu').detach().numpy()
            A_YP = yp[is_expr, 1, :].to('cpu').detach().numpy()
            D_YL = labels[is_expr, 2, :].to('cpu').detach().numpy()
            D_YP = yp[is_expr, 2, :].to('cpu').detach().numpy()

            donor_labels, donor_scores, acceptor_labels, acceptor_scores = get_donor_acceptor_scores(D_YL, A_YL, D_YP, A_YP)
            for idx in range(len(yp)):

                eles = seqname[idx].split(';')
                if len(eles) == 7:
                    chr, start, end, strand, name, aln_num, trans = eles
                    if strand == '+':
                        fw_junc_scores.write(f'{chr}\t{str(start)}\t{str(end)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\t{trans}\n')
                    elif strand == '-':
                        fw_junc_scores.write(f'{chr}\t{str(end)}\t{str(start)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\t{trans}\n')

                else:
                    chr, start, end, strand, name, aln_num = eles
                    if strand == '+':
                        fw_junc_scores.write(f'{chr}\t{str(start)}\t{str(end)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\n')
                    elif strand == '-':
                        fw_junc_scores.write(f'{chr}\t{str(end)}\t{str(start)}\t{name}\t{str(aln_num)}\t{strand}\t{str(donor_scores[idx])}\t{str(acceptor_scores[idx])}\n')
                
                junc_counter += 1            
            # increment the progress bar
            pbar.next()

    pbar.finish()
    fw_junc_scores.close()
    return out_score_f