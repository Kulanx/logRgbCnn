# Given a source directory and an image directory, 
# this program converts linear sRGB images to log images
# in exr 32-bit floating point format.

# image io requires an additional binary to
# deal with exr images.
# Find it here: https://imageio.readthedocs.io/en/v2.8.0/format_exr-fi.html.

import imageio
import cv2
import os
import numpy as np
from tqdm import tqdm
import sys

SRC_DIR_CATS = "data/lin/cat"
DST_DIR_CATS = "data/log/cat"
SRC_DIR_DOGS = "data/lin/cat"
DST_DIR_DOGS = "data/log/cat"

def conversion_linear_to_log(src_dir): 
    parent_dir = src_dir.split("/")[-2]
    print("parent_dir = " + parent_dir)
    animal_type = src_dir.split("/")[-1]
    dst_dir ="log" + "/" + parent_dir + "/" + animal_type
    is_exist = os.path.exists(dst_dir)
    if not is_exist:
        os.makedirs(dst_dir)

    PNG = '.png'
    files = os.listdir(src_dir)

    # imageio.plugins.freeimage.download()
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    for i in tqdm(range(len(files)), desc=f'Converting to log.'):
        item = files[i]
        if item[-4:] == PNG:
            path = os.path.join(src_dir, item)
            log_image = cv2.imread(path, cv2.IMREAD_UNCHANGED) #BGR
            if log_image is None:
                continue
            log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB) # convert to RGB from BGR before save
            # doing the resizing here (do not do resizing after data into log space)
            # The 224 sizes are used to allow for the extraction of random patches for translation invariance   
            # http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf 
            IMAGE_WIDTH=224
            IMAGE_HEIGHT=224
            log_image = cv2.resize(log_image, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
            # change from 8-bit to float 32
            log_image = log_image.astype("float32")
            log_image[log_image!=0] = np.log(log_image[log_image!=0]) # Do not take log of 0.
            imageio.imsave(os.path.join(dst_dir, f'{item[:-4]}.exr'), log_image) 

def main(argv):
    if len(argv) < 2:
        print("usage: python %s <original directory> <output directory>" % (argv[0]))
        return
    src_dir = argv[1].rstrip('/')

    conversion_linear_to_log(src_dir)

    print('done')

if __name__ == "__main__":
    main( sys.argv )
    print("** Terminating")