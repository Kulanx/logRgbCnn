import torch
from torch.utils.data import random_split
import torch.optim as optim
import CatOrDogNet
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from dataset_definitions.fish_log_image_dataset import FishLogImageDataset  
from dataset_definitions.fish_linear_image_dataset import FishLinearImageDataset  
from dataset_definitions.fish_jpg_image_dataset import FishJPGImageDataset
import matplotlib.pyplot as plt
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

parser = argparse.ArgumentParser(
                    prog='train',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('-i', '--input')
parser.add_argument('-m', '--model')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('-r', '--random', type=bool, default=False)
args = parser.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define the dataset and transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# image_root = os.path.join('processed', 'pseudolog')
image_type = args.input
print(f'image type = {image_type}')
print(f'model = {args.model}')
dataset = Dataset()
if args.input == 'log':
    image_root = os.path.join('processed', 'pseudolog')
    dataset = FishLogImageDataset(root_dir=image_root)
elif args.input == 'linear':
    image_root = os.path.join('processed', 'pseudolinear')
    dataset = FishLinearImageDataset(root_dir=image_root)
elif args.input == 'srgb':
    image_root = os.path.join('processed', 'srgb')
    dataset = FishJPGImageDataset(root_dir=image_root)
else:
    print('image nor recognized')
    raise Exception(f'--input {args.input} not recognized')

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
num_epochs = 40

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# change model here
net = models.densenet121()
if args.model.lower() == 'efficientnet':
    print('using efficient net')
    # change pretrained to false to initialize random weights
    net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
elif args.model.lower() == 'googlenet':
    print('using google net')
    net = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
elif args.model.lower() == 'mobilenetv3':
    net = models.mobilenet_v3_small(weights='DEFAULT')
# net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# net = CatOrDogNet.CatOrDogNet()
if torch.cuda.is_available():
    net.cuda()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataloader.dataset) for i in range(num_epochs + 1)]

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

def evaluate(x, test_losses, accuracy=[], title='Evaluate.png', x_label='number of epochs'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    ax[0].plot(x, test_losses, color='red', label='Test Loss')
    ax[0].set(xlabel=x_label, 
            ylabel='negative log likelihood loss',
            title='Losses')
    ax[0].legend()
    
    ax[1].plot(x, accuracy)
    ax[1].set(xlabel='number of epochs', ylabel='accuracy', 
              title='Accuracies')

    plt.savefig(title)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=args.lr)
# AdamW
# SGD: default

loss_lst = []
accuracy_lst = []
max_accuracy = 0
max_net = models.densenet121(pretrained=True)
for epoch in range(num_epochs):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device=device, dtype=torch.float), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % print_size == print_size - 1:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_size:.3f}')
        #     running_loss = 0.0

    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device=device, dtype=torch.float), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        loss_lst.append(loss.item())
        accuracy_lst.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            print(f'current max = {max_accuracy}')
            torch.save(net.state_dict(), f'results/net_{args.model}_{accuracy}_{loss_lst[-1]}.pth')
            torch.save(optimizer.state_dict(), f'results/opt_{args.model}_{accuracy}_{loss_lst[-1]}.pth')

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')

# export model here
# torch.save(max_net.state_dict(), 'results/cat_dog_model.pth')
# torch.save(optimizer.state_dict(), 'results/cat_dog_optimizer.pth')

print('Finished Training')
result = 'losses:\n' + str(loss_lst) + '\naccuracies:\n' + str(accuracy_lst)
print(result)
result_path = os.path.join('results', f'{args.input}_{args.model}.txt')
with open(result_path, "w") as output:
    output.write(result)

# plot model here
x_values = list(range(1, num_epochs + 1))
evaluate(x_values, loss_lst, accuracy=accuracy_lst)