import torchvision.transforms as T
from PIL import Image
from PIL import ImageCms
import os
import io
import cv2
import numpy as np

def srgb_to_linear_rgb(srgb):
  """Converts an sRGB image to linear RGB.

  Args:
    srgb: An sRGB image, represented as a NumPy array.

  Returns:
    A linear RGB image, represented as a NumPy array.
  """

  # Convert the sRGB image to a floating-point array.
  srgb = srgb.astype(np.float32)

  # Apply the sRGB inverse electro-optical transfer function (EOTF).
  linear_rgb = np.where(
      srgb <= 0.04045,
      srgb / 12.92,
      ((srgb + 0.055) / 1.055) ** 2.4
  )

  return linear_rgb

# generate input paths
image_type = "linear"
image_start = 0
image_end = 4000
image_count = image_end - image_start
transform = T.Resize([224, 224])

# input_folder = "rawdata"
input_folder = os.path.join("processed", "srgb")
output_folder = os.path.join("processed", image_type)
cat_folder = "Cat"
dog_folder = "Dog"

input_cat = os.path.join(input_folder, cat_folder)
input_dog = os.path.join(input_folder, dog_folder)
output_cat = os.path.join(output_folder, cat_folder)
output_dog = os.path.join(output_folder, dog_folder)

inputs = [input_cat, input_dog]
outputs = [output_cat, output_dog]
print(inputs)
print(outputs)

linear = True
failed = []

if linear:
    for folder_idx, cur_input in enumerate(inputs):
        for x in range(image_start, image_end + 1):
            cur_image_path = os.path.join(inputs[folder_idx], str(x) + '.jpg')
            output_image_path = os.path.join(outputs[folder_idx], str(x) + '.jpg')
            cur_image = cv2.imread(cur_image_path)
            if cur_image is not None:
                cur_image = srgb_to_linear_rgb(cur_image)
                cv2.imwrite(output_image_path, cur_image)
                print('linear saved ' + output_image_path)
            else:
                failed.append(cur_image_path)
                print('===================================================================== file does not exist ' + cur_image_path + ' =================================')
else:
    for folder_idx, cur_input in enumerate(inputs):
        for x in range(image_start, image_end + 1):
            cur_image_path = os.path.join(inputs[folder_idx], str(x) + '.jpg')
            output_image_path = os.path.join(outputs[folder_idx], str(x) + '.jpg')
            cur_image = cv2.imread(cur_image_path)
            if cur_image is not None:
                cur_image = cv2.resize(cur_image, (224, 224))
                cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path, cur_image)
                print('srgb saved ' + output_image_path)
            else:
                failed.append(cur_image_path)
                print('=====================================================================file does not exist ' + cur_image_path + '=================================')

print('failed images:')
print(failed)

