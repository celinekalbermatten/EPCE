import os
import shutil
import numpy as np
import sys

# set path to folder containingthe 4 data folders
data_path = sys.argv[1]

# ========================================
# Folder creation
# ========================================

sub_folders = ['train', 'test']
sub_sub_folders = ['HDR', 'LDR']

test_hdr = os.path.join(data_path, "raw_ref_video_hdr")
test_ldr = os.path.join(data_path, "raw_video_png")
train_hdr = os.path.join(data_path, "tile_ref_video_hdr")
train_ldr = os.path.join(data_path, "tile_ref_video_png")

image_folders = [test_hdr, test_ldr, train_hdr, train_ldr]

# create the corresponding folders
if not os.path.exists("./dataset"):
    print("Making final dataset directory")
    os.makedirs("./dataset")

for sub_folder in sub_folders:
    if not os.path.exists(f"./dataset/{sub_folder}"):
        print(f"Making {sub_folder} directory")
        os.makedirs(f"./dataset/{sub_folder}")
    
    for sub_sub_folder in sub_sub_folders:
        if not os.path.exists(f"./dataset/{sub_folder}/{sub_sub_folder}"):
            print(f"Making {sub_folder}/{sub_sub_folder} directory")
            os.makedirs(f"./dataset/{sub_folder}/{sub_sub_folder}")

# ===========================================
# Moving data into the corresponding folders
# ===========================================

for image_folder in image_folders:
    print(f"---------- Processing {image_folder} ----------")
    filenames = [fn for fn in os.listdir(image_folder)]
    print(f"{len(filenames)} files")

    count_test_hdr = 0
    count_test_ldr = 0
    count_train_hdr = 0
    count_train_ldr = 0

    for filename in filenames:
        src_path = os.path.join(image_folder, filename)

        if filename.endswith('_video.png'):
            dst_path = "./dataset/test/LDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_test_ldr += 1
        elif filename.endswith('_ref.hdr'):
            dst_path = "./dataset/test/HDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_test_hdr += 1
        elif not(filename.startswith(".")) and "_ref-" in filename and ".hdr" in filename:
            dst_path = "./dataset/train/HDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_train_hdr += 1
        elif not(filename.startswith(".")) and "_video-" in filename and ".png" in filename:
            dst_path = "./dataset/train/LDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_train_ldr += 1
        
    print(f"- test HDR images: {count_test_hdr}")
    print(f"- test LDR images: {count_test_ldr}")
    print(f"- train HDR images: {count_train_hdr}")
    print(f"- train LDR images: {count_train_ldr}")


filenames_3400 = [fn for fn in os.listdir("dataset/train/LDR")]
print(f"LDR train: {len(filenames_3400)}")
filenames_6800 = [fn for fn in os.listdir("dataset/train/HDR")]
print(f"HDR train: {len(filenames_6800)}")

print('All files have been moved accordingly!')
