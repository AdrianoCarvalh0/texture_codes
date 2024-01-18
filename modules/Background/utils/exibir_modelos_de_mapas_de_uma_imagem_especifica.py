from pathlib import Path
import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

#path linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

#path windows
#sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Funcoes_gerais import funcoes

from Background import background_generation as backgen

#root_dir linux
root_dir ="/home/adriano/projeto_mestrado/modules"

#root_dir windows
#root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

img_especifica = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@41-Image 3-20X'
pickle_dir = f'{root_dir}/Vessel_Models_pickle/'
array_pickles = funcoes.ler_diretorios(pickle_dir, img=img_especifica)

array_pickles_sorted = np.sort(array_pickles)

for i in range(len(array_pickles_sorted)):
  path = (pickle_dir + f'/{array_pickles_sorted[i]}')   
  arquivo_pickle = pickle.load(open(path, 'rb'))  
  vessel_map = arquivo_pickle['vessel_model'].vessel_map 
  mapa_original = vessel_map.mapped_values
  
  path_map = array_pickles_sorted[i].replace("[","").replace("]","").replace(".pickle","")  

  plt.figure(figsize=[15, 12])  
  plt.imshow(mapa_original, 'gray', vmin=0, vmax=255)

  # parte para salvar a imagem
  #plt.savefig(f'imagem_{i}.svg', format='svg')
  plt.plot()