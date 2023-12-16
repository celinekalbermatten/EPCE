import os
import random
import shutil

def reduce_dataset(dataset_path, output_path, percentage):
    """
    Reduce a given dataset by only taking a certain percentage of it
    """
    # define the train and test paths
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    # define the HDR and LDR paths
    hdr_path = 'HDR'
    ldr_path = 'LDR'

    # create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create the output directories
    reduced_train_path = os.path.join(output_path, 'train')
    reduced_test_path = os.path.join(output_path, 'test')
    os.makedirs(reduced_train_path, exist_ok=True)
    os.makedirs(reduced_test_path, exist_ok=True)
    

    def copy_files(input_folder, output_folder):
        """
        Copy a percentage of files
        """
        files = os.listdir(input_folder)
        num_files = len(files)
        num_to_copy = int(num_files * percentage)
        files_to_copy = random.sample(files, num_to_copy)
        for file in files_to_copy:
            shutil.copy2(os.path.join(input_folder, file), output_folder)
    
    # reduce the train set
    copy_files(os.path.join(train_path, hdr_path), os.path.join(reduced_train_path, hdr_path))
    copy_files(os.path.join(train_path, ldr_path), os.path.join(reduced_train_path, ldr_path))
    
    # reduce the test set
    copy_files(os.path.join(test_path, hdr_path), os.path.join(reduced_test_path, hdr_path))
    copy_files(os.path.join(test_path, ldr_path), os.path.join(reduced_test_path, ldr_path))

# apply the splitting
dataset_path = './dataset_final'
output_path = './dataset_final_reduced'
percentage_to_keep = 0.03  

reduce_dataset(dataset_path, output_path, percentage_to_keep)
