import torch
from torchvision import datasets as ds
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Processing import *

MNIST_ds = ds.MNIST("", train=True, download=True)

inputSize = 3 * 16
outputSize = 10

class NumberDet(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(inputSize, 32)
        self.hidden = nn.Linear(32, 16)
        self.output = nn.Linear(16, outputSize)

    def forward(self, x):
        x = fn.relu(self.input(x))
        x = fn.relu(self.hidden(x))
        x = self.output(x)
        return x

if __name__ == '__main__':
    splited_data = []
    for i in range(60000):
        splited_data.append(splitImage(MNIST_ds.data[i]))

    train_data = []
    for i in range(60000):
        features = []
        for item in splited_data[i]:
            features.append(processItem(item))
        train_data.append((torch.tensor(features), MNIST_ds[i][1]))
        print("Sample n°", i, " done!")

    myNN = NumberDet()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(myNN.parameters(), lr=0.002)
    epochs = 20

    train_ds, validation_ds = torch.utils.data.random_split(train_data, [50000, 10000])
    loaderTrain, loaderValidation = DataLoader(train_ds, batch_size=24), DataLoader(validation_ds, batch_size=24)
    # Boucle d'apprentissage
    for epoch in range(epochs):
        myNN.train()
        for features in loaderTrain:
            data, label = features
            output = myNN(data.float().view(-1, 48))
            cost = loss(output, label)
            myNN.zero_grad()
            cost.backward()
            optimizer.step()

        # Validation
        valid_loss, correct = 0, 0
        myNN.eval()
        for features in loaderValidation:
            data, label = features
            output = myNN(data.float().view(-1, 48))
            cost = loss(output, label)
            valid_loss += cost.item()
            correct += torch.sum(torch.argmax(output, dim=1) == label).item()
        valid_loss /= len(loaderValidation)
        correct /= len(loaderValidation.dataset)

        print(f"epoch: {epoch + 1}, validation loss: {valid_loss:.4f}, correct predictions: {correct * 100:.2f}%")

    torch.save(myNN.state_dict(), "Prediction_Model.pth")