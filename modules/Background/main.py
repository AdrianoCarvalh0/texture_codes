import pickle
import numpy as np
from PIL import Image
import skimage as ski
import sys, time
import geopandas as gpd
from matplotlib import pyplot as plt
from Image_properties import props

sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

from Funcoes_gerais import funcoes

import background_generation as backgen

root_dir = f"/home/adriano/projeto_mestrado/modules"
img_dir = f'{root_dir}/Imagens/vessel_data/images/'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x/'
pickle_dir = f'{root_dir}/Vessel_Models_pickle'
background_dir = f'{root_dir}/Background/Mapas_gerados_artificialmente'
tracados_dir = f'{root_dir}/Artificial_Lines/tracados_bezier'

vetor_pickles = funcoes.ler_diretorios(pickle_dir)
array_backgrounds = funcoes.ler_diretorios(background_dir)
array_tracados = funcoes.ler_diretorios(tracados_dir)
n_backgrounds = np.random.randint(0, len(array_backgrounds))

background = np.array(Image.open(f'{background_dir}/{array_backgrounds[n_backgrounds]}'))
fundo_com_vasos = background.copy()
possui_mapas =  np.full(shape = background.shape, fill_value=0)
problema = 0
for i in range(25):
   
    n_tracados = np.random.randint(0, len(array_tracados))
    tracado = array_tracados[n_tracados]

    vetor_medial_path = backgen.retorna_paths(f'{tracados_dir}/{tracado}')
    
    vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1 = backgen.inserir_vasos(vetor_medial_path[0],vetor_medial_path[1],vetor_pickles,pickle_dir,background)
    try:
        fundo_com_vasos = backgen.inserir_mapa(fundo_com_vasos,vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1, possui_mapas)
    except:
        problema += 1
    # i = i + 2

print(problema)
plt.figure(figsize=[10, 8])
plt.title("fundo_com_vasos")
plt.imshow(fundo_com_vasos, 'gray', vmin=0, vmax=255)
plt.plot()

img1 = Image.fromarray(fundo_com_vasos.astype(np.uint8))

img = img1.save("teste99.tiff")
