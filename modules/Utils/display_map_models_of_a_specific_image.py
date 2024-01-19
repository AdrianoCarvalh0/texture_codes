from pathlib import Path
import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# Path for Linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

# Path for Windows
# sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from modules.Utils import functions
from Background import background_generation as backgen

# Root directory for Linux
root_dir = "/home/adriano/projeto_mestrado/modules"

# Root directory for Windows
# root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Specific image name
specific_image = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@41-Image 3-20X'
pickle_directory = f'{root_dir}/Vessel_Models_pickle/'
array_pickles = functions.read_directories(pickle_directory, img=specific_image)

# Sort the array of pickle files
array_pickles_sorted = np.sort(array_pickles)

# Loop through the sorted array of pickle files
for i in range(len(array_pickles_sorted)):
    path = (pickle_directory + f'/{array_pickles_sorted[i]}')   
    pickle_file = pickle.load(open(path, 'rb'))  
    vessel_map = pickle_file['vessel_model'].vessel_map 
    original_map = vessel_map.mapped_values

    # Extracting the map name from the pickle file name
    map_name = array_pickles_sorted[i].replace("[", "").replace("]", "").replace(".pickle", "")  

    # Plotting the original map
    plt.figure(figsize=[15, 12])  
    plt.imshow(original_map, 'gray', vmin=0, vmax=255)

    # Save the image (uncomment the line below to save the images)
    # plt.savefig(f'image_{i}.svg', format='svg')
    plt.plot()
