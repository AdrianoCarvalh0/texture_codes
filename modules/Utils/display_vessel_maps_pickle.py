import sys, pickle
from matplotlib import pyplot as plt

sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

from modules.Utils import functions


root_dir = f"/home/adriano/projeto_mestrado/modules"

pickle_dir = f'{root_dir}/Vessel_Models_pickle/'

vector_pickles = functions.read_directories(pickle_dir)


for i in range(len(vector_pickles)):

    path = (pickle_dir + f'/{vector_pickles[i]}')
    file_pickle = pickle.load(open(path, 'rb')) 
    vessel_map = file_pickle['vessel_model'].vessel_map 
    original_map = vessel_map.mapped_values
    print(f'/{vector_pickles[i]}')
    plt.figure(figsize=[10, 8])
    plt.title("original_map")
    plt.imshow(original_map, 'gray', vmin=0, vmax=60)
    plt.plot()