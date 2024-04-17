import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from pronet import ProNet
from dataset_dataloader import HandsDataset 

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
    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

if __name__ == '__main__':
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")

    train_dataset = HandsDataset("train.csv", None, CUDA)  # Adjust path and normalization as necessary
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = HandsDataset("test.csv", None, CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = ProNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_dataloader, criterion, optimizer)
    evaluate(model, test_dataloader)
