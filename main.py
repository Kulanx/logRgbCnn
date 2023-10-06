import torch
from torch.utils.data import random_split
import torch.optim as optim
import CatOrDogNet
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

batch_size = 4

# Define the dataset and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='processed/linear', transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate the size of the training and testing sets
total_size = len(dataset)
train_size = int(0.75 * total_size)
test_size = total_size - train_size

# Create a random split of the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing
batch_size = 32
print_size = 10
num_epochs = 25

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

net = CatOrDogNet.CatOrDogNet()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_lst = []
accuracy_lst = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_size == print_size - 1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_size:.3f}')
            running_loss = 0.0

    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        loss_lst.append(loss.item())
        accuracy_lst.append(accuracy)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')

print('Finished Training')
result = 'losses:\n' + str(loss_lst) + '\naccuracies:\n' + str(accuracy_lst)
print(result)
result_path = os.path.join('results', 'result2.txt')
with open(result_path, "w") as output:
    output.write(result)
