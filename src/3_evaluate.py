# evaluate the model
import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch import nn
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, recall_score, precision_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from pronet import ProNet
from prodata import ProData, get_dataloader

### GLOBAL VARIABLES
MODELS_DIR = '../results/2/runs/experiment_2/models'
NUM_CLASSES = 25
train_datafile = '../results/1/train_dataloader.pt'
test_datafile = '../results/1/test_dataloader.pt'
fig_dir = '../results/fig'
output_file = '../results/3/data.tsv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
num_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Z'}


def evaluate_model(model_path, dataloader, class_count, basename, device='cuda', criterion=nn.CrossEntropyLoss()):
    # Load the model
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    # Storage for predictions and actuals
    y_preds = []
    y_true = []

    # Evaluation loop
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader), ncols=0, desc="Evaluating Data...", unit=" step")
        for batch_idx, data in enumerate(dataloader):

            seqs, labels, prot_id, cog_id = data
            seqs = seqs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            seqs = torch.permute(seqs, (0, 2, 1))

            pred = model(seqs)
            loss = criterion(pred, labels)

            outputs = torch.softmax(pred, dim=1)

            y_preds.append(outputs)
            y_true.append(labels)

            pbar.update(1)
        pbar.close()

    # Concatenate all batches
    print('-'*120)
    y_preds = torch.cat(y_preds, dim=0).cpu().numpy()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    print(y_preds, y_true)

    print('-'*120)

    # Binarize the labels for multi-class
    y_true_bin = label_binarize(y_true, classes=range(class_count))
    print(y_true_bin)
    print('-'*120)

    # Calculate metrics
    y_labels_class = np.argmax(y_true, axis=1)
    y_predictions_class = np.argmax(y_preds, axis=1)
    print(y_labels_class, y_predictions_class)
    print('-'*120)
    accuracy = accuracy_score(y_labels_class, y_predictions_class)
    precision_macro = precision_score(y_labels_class, y_predictions_class, average='macro')
    recall_macro = recall_score(y_labels_class, y_predictions_class, average='macro')

    print(accuracy, precision_macro, recall_macro)
    print('-'*120)
    
    # Get labels
    # class_nums = np.arange(np.max(y_predictions_class))
    # class_labels = [num_to_label[i] for i in class_nums]
    # print(class_nums)
    # print(class_labels)
    class_labels = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V']
    print(class_labels)
    print('-'*120)

    ### CONFUSION MATRIX
    cm = confusion_matrix(y_labels_class, y_predictions_class)
    cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(cm)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {basename}')
    plt.savefig(f'{fig_dir}/{basename}_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
    # plt.show()



    ### MULTI-CLASS ROC CURVE
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_count):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(15, 9))
    plt.figure(tight_layout=True)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, class_count)))
    for i, color in zip(range(class_count), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=0.8,
                 label=f'{num_to_label[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-Class Receiver Operating Characteristic for {basename}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.savefig(f'{fig_dir}/{basename}_Multi_Class_ROC_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()


    ### MULTI-CLASS PRECISION-RECALL CURVE
    precision = dict()
    recall = dict()
    average_precision = dict()

    plt.figure(figsize=(15, 9))
    plt.figure(constrained_layout=True)
    for i in range(class_count):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_preds[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_preds[:, i])
        plt.plot(recall[i], precision[i], lw=0.8, 
                label=f'{num_to_label[i]} (AP = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Multi-Class Precision-Recall for {basename}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.savefig(f'{fig_dir}/{basename}_Multi_Class_Precision_Recall.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return {'accuracy': accuracy, 'precision_macro': precision_macro, 'recall_macro': recall_macro}

### RUNNER
if __name__ == "__main__":

    device = 'cpu'
    if torch.cuda.is_available(): 
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    model_paths = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for name in files:
            model_paths.append(os.path.join(root, name))
    model_paths.sort()

    print('loading data...')
    #train_loader = torch.load(train_datafile)
    test_loader = torch.load(test_datafile)
    print('done loading data.')
    
    with open(output_file, 'w') as f:
        #f.write(f'Model\tTrain Accuracy\tTrain Precision (Macro)\tTrain Recall (Macro)\tTest Accuracy\tTest Precision (Macro)\tTest Recall (Macro)\n')
        f.write(f'Model\tTest Accuracy\tTest Precision (Macro)\tTest Recall (Macro)\n')
        for model_path in model_paths[2:3]:
            basename = os.path.basename(model_path)
            print(f'Running evaluation for {basename}...')
            #train_metrics = evaluate_model(model_path, train_loader, NUM_CLASSES, basename, device)   
            test_metrics = evaluate_model(model_path, test_loader, NUM_CLASSES, basename, device)
            #f.write(f'{basename}\t{train_metrics["accuracy"]}\t{train_metrics["precision_macro"]}\t{train_metrics["recall_macro"]}\t{test_metrics["accuracy"]}\t{test_metrics["precision_macro"]}\t{test_metrics["recall_macro"]}\n')
            f.write(f'{basename}\t{test_metrics["accuracy"]}\t{test_metrics["precision_macro"]}\t{test_metrics["recall_macro"]}\n')


# def get_accuracy(y_prob, y_true):
#     pred_lab = torch.argmax(y_prob,dim=1)
#     labels_lab = torch.argmax(y_true,dim=1)
#     total += labels_lab.size(0)
#     correct += (pred_lab == labels_lab).sum().item() 
#     return float(correct / total)

#     # assert y_true.ndim == 1 and y_true.size() == y_prob.size()
#     # y_prob = y_prob > 0.5
#     # return (y_true == y_prob).sum().item() / y_true.size(0)