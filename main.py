import torch
from torch.utils.data import random_split
import torch.optim as optim
import CatOrDogNet
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the dataset and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_root = '/Users/kulanx/Git/CV_Object_Recognition/lin/test_set';
# cats = load_images_from_folder(image_root + '/cat')
# dogs = load_images_from_folder(image_root + '/dog')
# image_dict = {'cat': cats, 'dog': dogs}
# dataset = torch.Tensor(image_dict)
dataset = datasets.ImageFolder(root=image_root, transform=transform)
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

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

net = models.resnet18(weights='weights=ResNet18_Weights.DEFAULT')
# net = CatOrDogNet.CatOrDogNet()
if torch.cuda.is_available():
    net.cuda()

n_epochs=10
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataloader.dataset) for i in range(n_epochs + 1)]

def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        accuracy))
    return accuracy

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(2):  # loop over the dataset multiple times
    bad_count=0
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        try:
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
        except:
            bad_count += 1

    test()


print('Finished Training')
