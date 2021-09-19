
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np


BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
EPOCHS = 10  

pipeline = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  
])


train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)

test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    file = f.read()
image1 = [int(str(item).encode('ascii'), 16) for item in file[16:16+784]]
print(image1)
image1_np = np.array(image1, dtype=np.uint8). reshape(28, 28, 1)
print(image1_np.shape)

cv2.imwrite("digit.jpg",image1_np)


class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  
        self.conv2 = nn.Conv2d(10, 20, 3)  
        self.fc1 = nn.Linear(20*10*10, 500)  
        self.fc2 = nn.Linear(500, 10)  

    def forward(self, x):
        input_size = x.size(0) 
        x = self.conv1(x)  
        x = F.relu(x)  
        x = F.max_pool2d(x, 2, 2)  

        x = self.conv2(x) 
        x = F.relu(x)

        x = x.view(input_size, -1)  

        x = self.fc1(x) 
        x = F.relu(x)  

        x = self.fc2(x) 

        output = F.log_softmax(x, dim=1)  

        return output



model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())


def train_model(model, device, train_loader, optimizer, epoch):
   
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
       
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
       
        output = model(data)
       
        loss = F.cross_entropy(output, target)
       
        loss.backward()
        
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss: {:.6f}".format(epoch, loss.item()))


def test_model(model, device, test_loader):
    
    model.eval()
    
    correct = 0.0
    
    test_loss = 0.0
    with torch.no_grad():  
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)
           
            output = model(data)
            
            test_loss += F.cross_entropy(output, target).item()
           
            pred = output.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test -- Average Loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100.0*correct/len(test_loader.dataset)))



for epoch in range(1,EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)
