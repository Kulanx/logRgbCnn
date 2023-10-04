import torchvision.transforms as T
from PIL import Image
from PIL import ImageCms
import os
import io

image_type = "srgbsmall"
image_start = 0
image_end = 200
image_count = image_end - image_start
transform = T.Resize(224)

input_folder = "rawdata"
output_folder = os.path.join("processed", image_type)
cat_folder = "Cat"
dog_folder = "Dog"

input_cat = os.path.join(input_folder, cat_folder)
input_dog = os.path.join(input_folder, dog_folder)
output_cat = os.path.join(output_folder, cat_folder)
output_dog = os.path.join(output_folder, dog_folder)

inputs = [input_cat, input_dog]
outputs = [output_cat, output_dog]


def convert_to_srgb(img):
    '''Convert PIL image to sRGB color space (if possible)'''
    if img.mode in ["RGBA", "P"]:
        img = img.convert("RGB")
    icc = img.info.get('icc_profile', '')
    if icc:
        io_handle = io.BytesIO(icc)     # virtual file
        src_profile = ImageCms.ImageCmsProfile(io_handle)
        dst_profile = ImageCms.createProfile('sRGB')
        img = ImageCms.profileToProfile(img, src_profile, dst_profile)
    img = transform(img)
    return img


for folder_idx, cur_input in enumerate(inputs):
    for x in range(image_start, image_end + 1):
        cur_image_path = os.path.join(inputs[folder_idx], str(x) + '.jpg')
        output_image_path = os.path.join(outputs[folder_idx], str(x) + '.jpg')
        cur_image = Image.open(cur_image_path)
        processed_image = convert_to_srgb(cur_image)
        processed_image.save(output_image_path)
        print('saved ' + output_image_path)
