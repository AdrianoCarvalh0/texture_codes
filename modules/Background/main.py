from pathlib import Path
import numpy as np
from PIL import Image
import sys
from matplotlib import pyplot as plt
from Funcoes_gerais import funcoes
import background_generation as backgen

#windows
#sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
#root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

#linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
root_dir = f"/home/adriano/projeto_mestrado/modules"


img_dir = f'{root_dir}/Imagens/vessel_data/images'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x'
training_dir = f'{root_dir}/Training_validation'

pickle_dir_5 = f'{training_dir}/Maps/5_maps_de_images'
pickle_dir_10 = f'{training_dir}/Maps/10_maps_10_images'
pickle_dir_40 = f'{training_dir}/Maps/160_maps_40_images'
pickle_dir_50 = f'{training_dir}/Maps/200_maps_50_images'
pickle_dir_343 = f'{root_dir}/Vessel_Models_pickle'

background_dir = f'{root_dir}/Background/Mapas_gerados_artificialmente'
background_dir_5 = f'{training_dir}/Backgrounds/5_backgrounds'
background_dir_10 = f'{training_dir}/Backgrounds/10_backgrounds'
background_dir_40 = f'{training_dir}/Backgrounds/40_backgrounds'
background_dir_50 = f'{training_dir}/Backgrounds/50_backgrounds'
