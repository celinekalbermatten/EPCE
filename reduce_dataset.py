import os
import random
import shutil

def reduce_dataset(dataset_path, output_path, percentage):
    hdr_extension = '.hdr'
    ldr_extension = '.png'
    
    def get_matching_pairs(input_folder_hdr, input_folder_ldr, folder):
        hdr_files = [file for file in os.listdir(input_folder_hdr) if file.endswith(hdr_extension)]
        ldr_files = [file for file in os.listdir(input_folder_ldr) if file.endswith(ldr_extension)]
        
        matching_pairs_train = []
        matching_pairs_test = []
        for hdr_file in hdr_files:
            hdr_prefix = hdr_file[:5]  # First 5 characters
            hdr_prefix_long = hdr_file[:9]
            hdr_suffix = hdr_file[-9:-4]  # Last 5 characters before extension
            
            for ldr_file in ldr_files:
                ldr_prefix = ldr_file[:5]  # First 5 characters
                ldr_prefix_long = ldr_file[:9]
                ldr_suffix = ldr_file[-9:-4]  # Last 5 characters before extension
                
                if folder == 'train':
                    if hdr_prefix == ldr_prefix and hdr_suffix == ldr_suffix:
                        matching_pairs_train.append((hdr_file, ldr_file))
                        break  # Found matching LDR file, break inner loop
                
                elif folder == 'test':
                    if hdr_prefix_long == ldr_prefix_long:
                        matching_pairs_test.append((hdr_file, ldr_file))
                        break
        
        if folder == 'train':
            return matching_pairs_train
        
        else:
            return matching_pairs_test
        
    
    def copy_pairs(pairs, input_folder_hdr, input_folder_ldr, output_folder_hdr, output_folder_ldr):
        num_pairs = len(pairs)
        num_to_copy = int(num_pairs * percentage)
        pairs_to_copy = random.sample(pairs, num_to_copy)
        for hdr, ldr in pairs_to_copy:
            shutil.copy2(os.path.join(input_folder_hdr, hdr), os.path.join(output_folder_hdr, hdr))
            shutil.copy2(os.path.join(input_folder_ldr, ldr), os.path.join(output_folder_ldr, ldr))
    
    for folder in ['train', 'test']:
        input_folder_hdr = os.path.join(dataset_path, folder, 'HDR')
        input_folder_ldr = os.path.join(dataset_path, folder, 'LDR')
        
        output_folder_hdr = os.path.join(output_path, folder, 'HDR')
        output_folder_ldr = os.path.join(output_path, folder, 'LDR')
        
        os.makedirs(output_folder_hdr, exist_ok=True)
        os.makedirs(output_folder_ldr, exist_ok=True)
        
        pairs = get_matching_pairs(input_folder_hdr, input_folder_ldr, folder)
        copy_pairs(pairs, input_folder_hdr, input_folder_ldr, output_folder_hdr, output_folder_ldr)



# apply the splitting
dataset_path = './dataset_final'
output_path = './dataset_final_reduced'
percentage_to_keep = 0.03  

reduce_dataset(dataset_path, output_path, percentage_to_keep)