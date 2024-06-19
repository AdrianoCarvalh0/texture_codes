import sys, pickle
from matplotlib import pyplot as plt
from pathlib import Path

# Linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

# Windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Utils import functions


pickle_dir = f'{root_dir}/Vessel_models_pickle/retina/10_maps'

#pickle_dir =  f'{root_dir}/Training_validation_retina/Maps/10_maps',  # directory of pickles

vector_pickles = functions.read_directories(pickle_dir)


for i in range(len(vector_pickles)):
#for i in range(1):

    path = (pickle_dir + f'/{vector_pickles[i]}')
    file_pickle = pickle.load(open(path, 'rb')) 
    vessel_map = file_pickle['vessel_model'].vessel_map 
    original_map = vessel_map.mapped_values
    
    #print(f'/{vector_pickles[i]}')
    plt.figure(figsize=[10, 8])
    plt.title(f"{vector_pickles[i]}")
    plt.imshow(original_map, 'gray', vmin=0, vmax=255)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')   
    plt.axis('off')
    #plt.savefig(f'img_{i}.svg', format='svg')