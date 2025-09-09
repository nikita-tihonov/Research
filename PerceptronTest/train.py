import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

batch_size = 64
num_classes = 10

train_dataset = datasets.MNIST(root='/mnist', 
                                        train = True,
                                        transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(1., ), std =(0.5, ))]),
                                        download = False)

test_dataset = datasets.MNIST(root='/mnist',
                                        train = False,
                                        transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(1.,), std = (0.5, ))]),
                                        download = False)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = batch_size,
                          shuffle = True)

test_loader = DataLoader(dataset = test_dataset,
                          batch_size = batch_size,
                          shuffle = True)

class Perceptron(nn.Module):
    def __init__(self, num_classes):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(784, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, num_classes)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        return out

model = Perceptron(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3

loss_train = []
loss_test = []

total_step = len(train_loader)
for epoch in range(num_epochs):
    sum_loss = 0
    n_batches = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        n_batches += 1
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    loss_train.append(sum_loss / n_batches)

    sum_loss = 0
    n_batches = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        sum_loss += loss
        n_batches += 1
    loss_test.append(sum_loss / n_batches)
    

loss_train = np.array([loss.item() for loss in loss_train])
loss_test = np.array([loss.item() for loss in loss_test])



x = [i for i in range(num_epochs)]

fig, ax = plt.subplots()

ax.plot(x, loss_train, color='blue')
ax.plot(x, loss_test, color='red')
plt.savefig('loss_plot.png')
plt.show()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


torch.save(model, 'perceptron.pth')