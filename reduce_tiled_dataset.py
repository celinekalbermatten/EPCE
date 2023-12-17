import os
import random
import shutil
import math

# set the seed for reproducibility
random.seed(1)

def reduce_training_dataset(dataset_path, output_path_train, output_path_test, train_percentage, test_percentage):
    """
    Reduces the size of a dataset by copying a specified percentage of images to new train and test folders.
    
    Parameters:
        dataset_path (str): Path to the original dataset folder.
        output_path_train (str): Path to the output folder for the reduced training dataset.
        output_path_test (str): Path to the output folder for the reduced testing dataset.
        train_percentage (float): Percentage of images to be included in the reduced training set.
        test_percentage (float): Percentage of images to be included in the reduced testing set.
    """
    hdr_extension = '.hdr'
    ldr_extension = '.png'
    
    def get_matching_pairs(input_folder_hdr, input_folder_ldr):
        """
        Retrieves pairs of HDR and LDR images that correspond to each other based on file naming conventions.
        
        Parameters:
            input_folder_hdr (str): Path to the HDR image folder within the dataset.
            input_folder_ldr (str): Path to the LDR image folder within the dataset.
        
        Returns:
            list: List of tuples containing matching pairs of HDR and LDR image filenames.
        """
        hdr_files = [file for file in os.listdir(input_folder_hdr) if file.endswith(hdr_extension)]
        ldr_files = [file for file in os.listdir(input_folder_ldr) if file.endswith(ldr_extension)]
        
        matching_pairs = []
        for hdr_file in hdr_files:
            # take the first 5 characters (number of the image)
            hdr_prefix = hdr_file[:5]  
            # take the last 5 characters (number of the image)
            hdr_suffix = hdr_file[-9:-4]  
            
            for ldr_file in ldr_files:
                # take the first 5 characters (number of the image)
                ldr_prefix = ldr_file[:5]  
                # take the last 5 characters (number of the image)
                ldr_suffix = ldr_file[-9:-4]  
                
                # ensure that the same images of hdr and ldr are taken
                if hdr_prefix == ldr_prefix and hdr_suffix == ldr_suffix:
                    matching_pairs.append((hdr_file, ldr_file))
                    break
        
        return matching_pairs
    
    def copy_images(pairs, input_folder_hdr, input_folder_ldr, output_folder_hdr, output_folder_ldr, num_to_copy):
        """
        Copies a specified number of selected image pairs from the original dataset to the output folders.
        
        Parameters:
            pairs (list): List of tuples representing image pairs to be copied.
            input_folder_hdr (str): Path to the HDR image folder in the original dataset.
            input_folder_ldr (str): Path to the LDR image folder in the original dataset.
            output_folder_hdr (str): Path to the HDR output folder for the reduced dataset.
            output_folder_ldr (str): Path to the LDR output folder for the reduced dataset.
            num_to_copy (int): Number of image pairs to copy to the output folders.
        """
        random.shuffle(pairs)
        selected_pairs = pairs[:num_to_copy]
        for hdr, ldr in selected_pairs:
            shutil.copy2(os.path.join(input_folder_hdr, hdr), os.path.join(output_folder_hdr, hdr))
            shutil.copy2(os.path.join(input_folder_ldr, ldr), os.path.join(output_folder_ldr, ldr))
    
    output_path_train_hdr = os.path.join(output_path_train, 'HDR')
    output_path_train_ldr = os.path.join(output_path_train, 'LDR')
    
    output_path_test_hdr = os.path.join(output_path_test, 'HDR')
    output_path_test_ldr = os.path.join(output_path_test, 'LDR')
    
    os.makedirs(output_path_train_hdr, exist_ok=True)
    os.makedirs(output_path_train_ldr, exist_ok=True)
    os.makedirs(output_path_test_hdr, exist_ok=True)
    os.makedirs(output_path_test_ldr, exist_ok=True)
    
    pairs_train = get_matching_pairs(os.path.join(dataset_path, 'train', 'HDR'), os.path.join(dataset_path, 'train', 'LDR'))
    total_images_train = len(pairs_train)
    
    num_train_images = math.ceil(total_images_train * train_percentage)
    num_test_images = math.ceil(total_images_train * test_percentage)
    
    copy_images(pairs_train, os.path.join(dataset_path, 'train', 'HDR'), os.path.join(dataset_path, 'train', 'LDR'), output_path_train_hdr, output_path_train_ldr, num_train_images)
    
    remaining_pairs_train = pairs_train[num_train_images:]
    copy_images(remaining_pairs_train, os.path.join(dataset_path, 'train', 'HDR'), os.path.join(dataset_path, 'train', 'LDR'), output_path_test_hdr, output_path_test_ldr, num_test_images)


# create the directory if it doesn't exist
if not os.path.exists('./dataset_final_reduced'):
    os.makedirs('./dataset_final_reduced')

# Apply the function to create the reduced dataset with specific percentages
dataset_path = './dataset_final'
output_path_train = './dataset_final_reduced/train'
output_path_test = './dataset_final_reduced/test'
# 2% for reduced training set
train_percentage = 0.02  
# 0.3% for reduced testing set
test_percentage = 0.003  

reduce_training_dataset(dataset_path, output_path_train, output_path_test, train_percentage, test_percentage)