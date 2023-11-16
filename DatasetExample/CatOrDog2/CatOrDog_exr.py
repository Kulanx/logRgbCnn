import torch
from torch import nn
import random
from PIL import Image
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from ImageClassifier import *
from torchinfo import summary
from tqdm.auto import tqdm
from CatDogLogImageSubDataset import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
sns.set_theme()

device = "cuda" if torch.cuda.is_available() else "cpu"

def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {Path(image_path).parent.stem}", fontsize=16)
    
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def plot_loss_curves(results):
  
    results = dict(list(results.items()))

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()



def main():
    image_path = "data"
    walk_through_dir(image_path)

    train_dir = "/Users/kexinhao/Desktop/code/CatOrDog/training_set/log/"
    test_dir = "/Users/kexinhao/Desktop/code/CatOrDog/test_set/log/"
    train_dir, test_dir

    # Set seed
    random.seed(42) 

    IMAGE_WIDTH=128
    IMAGE_HEIGHT=128
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    # Write transform for image
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # plot_transformed_images(image_path_list, transform=data_transform, n=3)

    # Creating training set
    # for linear
    train_data = CatDogLogImageSubDataset(root_dir=train_dir, custom_transforms=data_transform) # transforms to perform on labels (if necessary)
    #Creating test set
    test_data = CatDogLogImageSubDataset(root_dir=test_dir, custom_transforms=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

    # Get class names as a list
    class_names = train_data.class_map.keys()
    print("Class names: ",class_names)

    # Can also get class names as a dict
    class_dict = train_data.class_map
    print("Class names as a dict: ",class_dict)

    # Check the lengths
    print("The lengths of the training and test sets: ", len(train_data), len(test_data))

    print(train_data[0])
    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    # Rearrange the order of dimensions 
    img_permute = img.permute(1, 2, 0)

    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    plt.figure(figsize=(10, 7))
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    print(type(class_names))
    plt.title(list(class_names)[label], fontsize=14)

    # NUM_WORKERS = os.cpu_count()
    # print("workers:", NUM_WORKERS)

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=1, # how many samples per batch?
                                # num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data,
                                # num_workers=NUM_WORKERS,
                                batch_size=1, 
                                shuffle=False) # don't usually need to shuffle testing data

    train_dataloader, test_dataloader

    img, label = next(iter(train_dataloader))

    # Note that batch size will now be 1.  
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    # Set image size.
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    # Turn image folders into Datasets
    train_data_augmented = CatDogLogImageSubDataset(train_dir, custom_transforms=train_transform)
    test_data_augmented = CatDogLogImageSubDataset(test_dir, custom_transforms=test_transform)

    train_data_augmented, test_data_augmented

    # Set some parameters.
    BATCH_SIZE = 32
    torch.manual_seed(42)

    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=BATCH_SIZE,
                                            # num_workers=NUM_WORKERS,
                                            shuffle=True)

    test_dataloader_augmented = DataLoader(test_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        # num_workers=NUM_WORKERS,
                                        shuffle=False)

    train_dataloader_augmented, test_dataloader_augmented

    # Instantiate an object.
    model = ImageClassifier().to(device)

    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(train_dataloader_augmented))
    print(type(img_batch), label_batch)

    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device))
        
    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

    # do a test pass through of an example input size 
    summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT]) 

    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 25

    # Setup loss function and optimizer (hyperparameters)
    loss_fn = nn.CrossEntropyLoss()
    # talk to heather about this, how to set up the schedular
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3) 
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2, eps=1e-2) # adjust the learning rate, changing sigma to sth else might help, add scheduler
    # already tested - higher lr (1e-2 or 0.1) does not help (stop learning)
    # change the augmentation params might help (like flipping, translation, rotation, scaling, change brightness, adding noise)
    # for reference
    # https://datahacker.rs/016-pytorch-three-hacks-for-improving-the-performance-of-deep-neural-networks-transfer-learning-data-augmentation-and-scheduling-the-learning-rate-in-pytorch/

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_0 
    model_results = train(model=model,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_augmented,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    plot_loss_curves(model_results)

    data_type = input("what is the type of data? jpg, linear, or log. ")
    torch.save(model.state_dict(), data_type + '_model.pt')

if __name__ == '__main__':
    main()