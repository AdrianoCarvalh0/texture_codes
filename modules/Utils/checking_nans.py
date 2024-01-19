from pathlib import Path
import pickle
import numpy as np
import sys
from matplotlib import pyplot as plt

# Path for Linux
# sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

# Path for Windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from modules.Utils import functions

# Windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Linux
# root_dir = f"/home/adriano/projeto_mestrado/modules"

img_dir = f'{root_dir}/Images/vessel_data/images'
lab_dir = f'{root_dir}/Images/vessel_data/labels_20x'
train_dir = f'{root_dir}/Training_validation'

pickle_dir = f'{root_dir}/Vessel_Models_pickle'

pickle_dir_50 = f'{train_dir}/Maps/200_maps_of_50_images'

pickle_vector = functions.read_directories(pickle_dir_50)

dictionary_vector = []

for i in range(len(pickle_vector)):
    
    pickle_path = (pickle_dir_50 + f'/{pickle_vector[i]}')
    print(pickle_path)

    pickle_file = pickle.load(open(pickle_path, 'rb')) 
    vessel_map = pickle_file['vessel_model'].vessel_map 
    original_map = vessel_map.mapped_values   
    
    min_row = (np.min(np.rint(vessel_map.path2_mapped))-1)
    max_row  = (np.max(np.rint(vessel_map.path1_mapped))+1)
       
    dictionary = {
        'min_row': min_row,
        'max_row': max_row,
        'pickle': pickle_vector[i],
    }

    dictionary_vector.append(dictionary)


    # print(f'Minimum row: {min_row}')
    # print(f'Maximum row: {max_row}')

for array in dictionary_vector:
     if np.isnan(array['linha_minima']):
        print(array)

for array in dictionary_vector:
     if np.isnan(array['linha_maxima']):
        print(array)
