# Given a source directory and an destination directory, 
# this program inverse image from jpg sRGB to linear sRGB in png format

# checking file type example:
# from (original):
# (base) kexinhao@Kexins-MBP CatOrDog % magick identify training_set/jpg/cats/cat.1.jpg
# training_set/jpg_sRGB/original_cats/cat.1.jpg JPEG 300x280 300x280+0+0 8-bit sRGB 16880B 0.000u 0:00.001
# to (converted):
# (base) kexinhao@Kexins-MBP CatOrDog % magick identify training_set/lin/cats/cat.1.png   
# training_set/png_linear_sRGB/linear_cats/cat.1.png PNG 300x280 300x280+0+0 8-bit sRGB 128498B 0.000u 0:00.000

import os
from tqdm import tqdm
import sys

def conversion_jpg_to_linear(src_dir):
    
    parent_dir = src_dir.split("/")[-2]
    print("src dir = " + src_dir)
    animal_type = src_dir.split("/")[-1]
    dst_dir = "lin" + "/" + parent_dir + "/" + animal_type
    is_exist = os.path.exists(dst_dir)
    if not is_exist:
        os.makedirs(dst_dir)

    for filename in tqdm(os.listdir(src_dir), desc=f'Converting to linear.'):
        if filename.endswith('.jpg'):
            # use magick for conversion 
            os.system("magick convert {}/{} -colorspace RGB {}/{}.png".format(src_dir, filename, dst_dir, os.path.splitext(filename)[0]))

def main(argv):
    if len(argv) < 2:
        print("usage: python %s <original directory> <output directory>" % (argv[0]))
        return
    src_dir = argv[1].rstrip('/')

    conversion_jpg_to_linear(src_dir)

    print('done')

if __name__ == "__main__":
    main( sys.argv )
    print("** Terminating")

