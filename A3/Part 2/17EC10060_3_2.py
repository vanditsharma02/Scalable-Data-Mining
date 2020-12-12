from __future__ import print_function
import torch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

batch_size = 64

kwargs = {'batch_size': 64}
if use_cuda:
  kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('./', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('./', train=False,
                       transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

images, labels = next(iter(train_loader))
plt.imshow(images[43].reshape(28,28), cmap="gray")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #in_channel x out_channel x kernel_size x stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(accuracy)
    return test_loss

def main():
    nepochs = 5 # Can change it
    train_loss = []
    epoch_loss = []
    test_loss = []
    epochs = []
    model = Net()
    # UNCOMMENT TO CHOOSE OPTIMIZER
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    #optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    #optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)
    #optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.1)
    for epoch in range(0,nepochs):
        train_loss.extend(train(model, device, train_loader, optimizer, epoch))
        epoch_loss.append(train_loss[-1])
        test_loss.append(test(model, device, test_loader))
        epochs.append(epoch)
        #keep saving the model for which test_loss is the least and use it for final reporting
        if(epoch==0):
            torch.save(model.state_dict(), 'checkpoint.pth')
            prevloss = test_loss[-1]
        else:
            if(prevloss>test_loss[-1]):
                torch.save(model.state_dict(), 'checkpoint.pth')
                prevloss = test_loss[-1]

    plt.figure()
    plt.plot(train_loss)
    plt.title('Training Loss vs Updates')
    plt.xlabel('Update number')
    plt.ylabel('Training Loss')
    plt.show()
    plt.plot(epoch_loss)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()

    # SAMPLE CODE FOR PLOTTING ACCURACY
	# accuracy = [98.45, 98.66, 98.83, 11.35]
	# loc = np.arange(4)
	# plt.bar(loc, accuracy)
	# plt.ylabel('Test Accuracy')
	# plt.title('Test Accuracy vs Optimizer')
	# plt.xticks(loc, ('Mini Batch', 'Momentum', 'Nesterov', 'Adam'))
	# plt.show()

if __name__=='__main__':
    main()