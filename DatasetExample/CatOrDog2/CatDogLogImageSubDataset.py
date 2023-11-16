import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # Required to read exr images


class CatDogLogImageSubDataset(Dataset):
    """
    Standard torchvision transforms may not support exr images.
    LogImageDataset, a substitute of ImageFolder
    Defining custom datasets allows us to process jpg, png, and exr images in the same way.
    Output of this dataset is a float 32 torch tensor and a label for each image.
    Scaling has proven impractical for log images.

    Must enable:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    in CatOrDog_exr.py file to allow opencv to read exr images.
    """

    def __init__(self, root_dir, custom_transforms=[torch.tensor] ): # Default to convert to tensor. Otherwise, can't load data set.
        """
        Args:
            root_dir (string): Path to the root log image folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.df = self._build_dataset_df()
        self.class_map = {'cats' : 0, 'dogs' : 1}
        self.custom_transforms = custom_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0],
                                self.df.iloc[idx, 1])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Flag is important.
        if image is None:
            print(img_path)
        # image = image - 5.545 # Center the images on 0 - should not have this line of code, stop learning if have this
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.iloc[idx, 0]

        if self.custom_transforms:
          if (isinstance(self.custom_transforms, list)):
            for transform in self.custom_transforms:
                image = transform(image)
          else:
                image = self.custom_transforms(image)

        
        class_id = torch.tensor(self.class_map[label])
        image = torch.permute(image, (0, 2, 1))

        return image, class_id

    def _build_dataset_df(self):
      labels = os.listdir(self.root_dir)
      data = {'label': [], 'fname': []}
      for label in labels:
        # might have hidden files ".DS_Store"
        if (label != ".DS_Store"):
            for f in os.listdir(os.path.join(self.root_dir, label)):
                data['label'].append(label)
                data['fname'].append(f)
      return pd.DataFrame.from_dict(data)