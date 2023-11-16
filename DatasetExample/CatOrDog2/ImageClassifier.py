from torch import nn
import torch

# # Creating a CNN-based image classifier.
class ImageClassifier(nn.Module):
    """
    This work is based on https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
    """
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 512, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=5, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2),
          nn.Dropout2d(p=0.3)) # regularization, dropout rate over 0.3 would lead to lower train and test acc
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*5*5, out_features=64),
          nn.ReLU(),
          nn.Linear(in_features=64, out_features=2))
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        # x = self.conv_layer_3(x) - too much complexity in the feature extract, not too much complexity in the classifier, so removed one layer
        x = self.classifier(x)
        return x
