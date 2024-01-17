import pickle, os
import numpy as np
import sys
import skimage as ski
from pathlib import Path
from PIL import Image

from matplotlib import pyplot as plt

sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

import geopandas as gpd
import background_generation as backgen

sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Funcoes_gerais import funcoes

#root_dir = f"/home/adriano/projeto_mestrado/modules"
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

#dir linux
#root_dir ="/home/adriano/projeto_mestrado/modules"

img_dir = f'{root_dir}/Imagens/vessel_data/images'

imag = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 3-20X'

back_path = 'T-3 Weeks@Females@919 F@919-CTL-top-20X-01'

#Lendo o pickle e gerando o vessel_map
idx = 1
pickle_dir = f'{root_dir}/Vessel_Models_pickle'
path = (pickle_dir + f'/{imag}_savedata1.pickle')
arquivo = pickle.load(open(path, 'rb'))
vessel_map = arquivo['vessel_model'].vessel_map
mapa_original = vessel_map.mapped_values

rows, cols = mapa_original.shape[0], mapa_original.shape[1]

altura = (rows/2) 

arquivo = f'{root_dir}/Artificial_Lines/tracados_bezier_maiores/img_savedata_13.json'

background_path = f'{root_dir}/Background/Mapas_gerados_artificialmente/{back_path}.tiff'
background = np.array(Image.open(f'{background_path}'))

medial_path = backgen.retorna_paths(arquivo)   

mapa_expandido_original = backgen.expandir_mapas_do_tamanho_do_tracado(mapa_original,medial_path[1])

imagem_binaria_original = vessel_map.mapped_mask_values 
imagem_binaria_sem_artefatos_laterais = backgen.retornar_imagem_binaria_sem_artefatos(vessel_map, imagem_binaria_original) 
  
imagem_binaria_sem_artefatos = backgen.fill_holes(imagem_binaria_sem_artefatos_laterais) 
  
mapa_original_norm = backgen.normaliza(background,mapa_original,imagem_binaria_sem_artefatos,treshold=40)

distancia = int(medial_path[1])

vaso_expandido_bin = backgen.expandir_mapas_do_tamanho_do_tracado(imagem_binaria_sem_artefatos,distancia)       
    
linha_offset_esquerda, linha_central, linha_offset_direita, maior_tamanho = backgen.retorna_linhas_offset_posicao_tamanho(medial_path[0],altura)  

#Criação das linhas à direita, centro e à esquerda a partir do traçado originado pelas curvas de beizier
dst_array_np = backgen.retorna_dst_array_np(linha_offset_esquerda, linha_central,linha_offset_direita, maior_tamanho)

#Execução do algoritmo que faz a transformação do mapa expandido
img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = backgen.rotacionando_mapa_expandido(mapa_expandido_original,dst_array_np,maior_tamanho)

tri = tform_out._tesselation 
tri_inv = tform_out._inverse_tesselation

plt.figure(figsize=[100,80])
ax = plt.subplot(121)
plt.imshow(img_proper, 'gray')
x, y = tri.points.T
ax.plot(x, y, 'o')
ax.triplot(x, y, tri.simplices.copy())

ax = plt.subplot(122)
plt.imshow(img_out, 'gray')
x, y = tri_inv.points.T
ax.plot(x, y, 'o')
ax.triplot(x, y, tri_inv.simplices.copy())

plt.savefig('delaunay.svg',format='svg')