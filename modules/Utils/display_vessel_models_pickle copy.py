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

from Slice_mapper import slice_mapper


pickle_dir = f'{root_dir}/Vessel_models_pickle/'

vector_pickles = functions.read_directories(pickle_dir)
root_img = f'{root_dir}/Images/vessel_data/images'

imag = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 1-20X'

img = f'{root_dir}/{root_img}/{imag}.tiff'

path = (pickle_dir + f'/{imag}_savedata0.pickle')
file_pickle = pickle.load(open(path, 'rb')) 
vessel_map = file_pickle['vessel_model'].vessel_map 

vessel_model = file_pickle['vessel_model']

fig = plt.figure(figsize=[12, 10])
ax = fig.add_subplot()
slice_mapper.plot_model(img, vessel_model, cross_paths, ax)        
norm = ax.images[0].norm
norm.vmin, norm.vmax = 0, 60

original_map = vessel_map.mapped_values
print(f'/{vector_pickles[i]}')
plt.figure(figsize=[10, 8])
plt.title("original_map")
plt.imshow(original_map, 'gray', vmin=0, vmax=60)
plt.plot()

    