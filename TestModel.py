from Processing import *
from Model import DataLoader, NumberDet, torch, ds, nn

Test_ds = ds.MNIST("", train=False, download=True)

splited_data = []
for i in range(10000):
    splited_data.append(splitImage(Test_ds.data[i]))

test_data = []
for i in range(10000):
    features = []
    for item in splited_data[i]:
        features.append(processItem(item))
    test_data.append((torch.tensor(features), Test_ds[i][1]))
    print("Test sample n°", i, " done!")

loaderTest = DataLoader(test_data, batch_size=24)

if __name__ == '__main__':
    mySavedNN = NumberDet()
    mySavedNN.load_state_dict(torch.load("Prediction_Model.pth"))
    loss = nn.CrossEntropyLoss()
    for epoch in range(10):
        # Test
        valid_loss, correct = 0, 0
        mySavedNN.eval()
        for features in loaderTest:
            data, label = features
            output = mySavedNN(data.float().view(-1, 48))
            cost = loss(output, label)
            valid_loss += cost.item()
            correct += torch.sum(torch.argmax(output, dim=1) == label).item()
        valid_loss /= len(loaderTest)
        correct /= len(loaderTest.dataset)

        print(f"epoch: {epoch + 1}, test loss: {valid_loss:.4f}, correct predictions: {correct * 100:.2f}%")