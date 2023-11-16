import torch
from torch.utils.data import random_split
import torch.optim as optim
import CatOrDogNet
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
<<<<<<< HEAD
=======
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
>>>>>>> Add provided code.

batch_size = 4

# Define the dataset and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

<<<<<<< HEAD
dataset = datasets.ImageFolder(root='processed/linear', transform=transform)
=======
image_root = '/Users/kulanx/Git/CV_Object_Recognition/lin/test_set';
# cats = load_images_from_folder(image_root + '/cat')
# dogs = load_images_from_folder(image_root + '/dog')
# image_dict = {'cat': cats, 'dog': dogs}
# dataset = torch.Tensor(image_dict)
dataset = datasets.ImageFolder(root=image_root, transform=transform)
>>>>>>> Add provided code.
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
<<<<<<< HEAD

loss_lst = []
accuracy_lst = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

=======
for epoch in range(2):  # loop over the dataset multiple times
    bad_count=0
>>>>>>> Add provided code.
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
