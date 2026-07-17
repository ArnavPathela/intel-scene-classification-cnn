import torch 
import torch.nn as nn 
import torchvision
from torchvision.transforms import v2
from torchvision import datasets  
import torch.nn.functional as F
import matplotlib.pyplot as plt


train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((64,64)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(15),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((64,64)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size = 32

trainset = datasets.ImageFolder(root='seg_train/seg_train', transform=train_transform)
testset = datasets.ImageFolder(root='seg_test/seg_test', transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = trainset.classes
print(classes)



class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(2704,1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100,6)

    def forward(self,value):
        c1 = F.relu(self.conv1(value))
        s1 = F.max_pool2d(c1,(2,2))
        c2 = F.relu(self.conv2(s1))
        s2 = F.max_pool2d(c2,(2,2))
        s3 = torch.flatten(s2,1)
        s4 = F.relu(self.fc1(s3))
        s5 = F.relu(self.fc2(s4))

        output = self.fc3(s5)

        return output 
    
net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


epoch_loss = []
for epoch in range(20):
    running_loss = 0
    epoch_total_loss = 0
    num_batches = 0
    for i , data in enumerate(trainloader, 0):
        input, label = data
        
        optimizer.zero_grad()

        output = net(input)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()
        running_loss+= loss.item()
        epoch_total_loss += loss.item()   
        num_batches += 1  

        if i %100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0
    epoch_loss.append(epoch_total_loss / num_batches)

print("finished training")

plt.figure()
plt.plot(range(1, len(epoch_loss) + 1), epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss over Epochs')
plt.savefig('loss_curve.png')
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds = []
all_labels = []
            
correct = 0 
total = 0 

with torch.no_grad():
    for data in testloader:
        image, label = data

        output = net(image)

        _,predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        all_preds.extend(predicted.tolist())
        all_labels.extend(label.tolist())

print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()





