import sys, pickle, os
import tracemalloc, time
import numpy as np
from PIL import Image
sys.path.insert(0, "C:\\Users\\adria\\Documents\\Mestrado\\texture_codes\\modules")

import background_generation as backgen

root_dir ='"C:\\Users\\adria\\Documents\\Mestrado\\texture_codes\\modules"'

img_dir = root_dir + '\\Imagens\\vessel_data\\images/'
lab_dir = root_dir + '\\Imagens\\vessel_data\\labels_20x/'

#Varrendo o diretório
filenames = []
for filename in os.listdir(img_dir):
#     # Use only images having magnification 20x
#     #if 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers' in filename:
#     if 'T-3 Weeks' in filename:      
  filenames.append(filename.split('.')[0])
filenames = filenames[:20]

#Lendo o pickle e gerando o vessel_map
idx = 1
pickle_dir = f'{root_dir}\\Vessel_Models_pickle'
path = (pickle_dir + f'{filenames[idx]}_savedata1.pickle')
arquivo = pickle.load(open(path, 'rb'))
vessel_map = arquivo['vessel_model'].vessel_map
mapa_original = vessel_map.mapped_values

#Lendo o Json
#arquivo = '/content/drive/MyDrive/Mestrado em Ciência da Computação/Artificial Lines/arquivo_quatro_pontos.json'
arquivo = '{root_dir}\\Artificial Lines\\teste_31_05.json'
#arquivo = '/content/drive/MyDrive/Mestrado em Ciência da Computação/Artificial Lines/teste2_31_05.json'
#arquivo = '/content/drive/MyDrive/Mestrado em Ciência da Computação/Artificial Lines/teste.json'

medial_path = backgen.retorna_paths(arquivo)

caminhos_transladados = backgen.retorna_caminhos_transladados(medial_path)
caminhos_transladados_interpolado = backgen.retorna_caminhos_transladados(medial_path[0], 2)

vetor_pontos = backgen.construindo_caminhos(caminhos_transladados[0])
vetor_pontos_interp = backgen.construindo_caminhos(caminhos_transladados_interpolado[0])

linha_central,linha_offset_esquerda,linha_offset_direita, maior_tamanho = backgen.retorna_linhas_offset_posicao_tamanho(mapa_original,caminhos_transladados_interpolado[0])

dst_array_np = backgen.retorna_dst_array_np(linha_central,linha_offset_esquerda,linha_offset_direita, maior_tamanho)
