from pathlib import Path
import pickle
import numpy as np
import sys
from matplotlib import pyplot as plt

#path linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

#path windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Funcoes_gerais import funcoes


root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
img_dir = f'{root_dir}/Imagens/vessel_data/images'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x'
trein_dir = f'{root_dir}/Treinamento_validacao'

pickle_dir = f'{root_dir}/Vessel_Models_pickle'
vetor_pickles = funcoes.ler_diretorios(pickle_dir)

vetor_dict = []

for i in range(len(vetor_pickles)):
    
    path_pickle = (pickle_dir + f'/{vetor_pickles[i]}')
    print(path_pickle)

    arquivo_pickle = pickle.load(open(path_pickle, 'rb')) 
    vessel_map = arquivo_pickle['vessel_model'].vessel_map 
    mapa_original = vessel_map.mapped_values   
    
    linha_minima = (np.min(np.rint(vessel_map.path2_mapped))-1)
    linha_maxima  = (np.max(np.rint(vessel_map.path1_mapped))+1)
       
    dict ={
        'linha_minima':linha_minima,
        'linha_maxima':linha_maxima,
        'pickle': vetor_pickles[i],
    }

    vetor_dict.append(dict)


    #print(f'linha minima: {linha_minima}')
    #print(f'linha máxima: {linha_maxima}')