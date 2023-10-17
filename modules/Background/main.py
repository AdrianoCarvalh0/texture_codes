import numpy as np
from PIL import Image
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

from Funcoes_gerais import funcoes

import background_generation as backgen

root_dir = f"/home/adriano/projeto_mestrado/modules"
img_dir = f'{root_dir}/Imagens/vessel_data/images'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x'

trein_dir = f'{root_dir}/Treinamento_validacao'

pickle_dir = f'{root_dir}/Vessel_Models_pickle'
pickle_dir_10 = f'{trein_dir}/1_mapa_de_10_imagens_treinamento'
pickle_dir_40 = f'{trein_dir}/4_mapas_de_40_imagens_treinamento'

background_dir = f'{root_dir}/Background/Mapas_gerados_artificialmente'
background_dir_10 = f'{trein_dir}/10_backgrounds'
background_dir_40 = f'{trein_dir}/40_backgrounds'

tracados_dir = f'{root_dir}/Artificial_Lines/tracados_bezier'
tracados_dir_maiores = f'{root_dir}/Artificial_Lines/tracados_bezier_maiores'

vetor_pickles = funcoes.ler_diretorios(pickle_dir_10)
array_backgrounds = funcoes.ler_diretorios(background_dir_10)
array_tracados = funcoes.ler_diretorios(tracados_dir)
array_tracados_maiores = funcoes.ler_diretorios(tracados_dir_maiores)

problema = 0
resultados_not_none = 0
resultados_none = 0

for j in range(57):
    n_backgrounds = np.random.randint(0, len(array_backgrounds))
    background = np.array(Image.open(f'{background_dir_10}/{array_backgrounds[n_backgrounds]}'))
    nome_background = f'{array_backgrounds[n_backgrounds]}'
    background_recortado = background[0:1100,0:1370]

    nome_background = f'{array_backgrounds[n_backgrounds]}'

    nome_background = nome_background.replace("'","").replace(".tiff","")    

    background_com_pad = np.pad(background_recortado, ((200,200),(200,200)), mode="symmetric", reflect_type="even")
    
    fundo_com_vasos = background_com_pad.copy()
    
    possui_mapas =  np.full(shape = background_com_pad.shape, fill_value=0)
    n_vasos = np.random.randint(20, 50)    
    contador = 0
    while contador < n_vasos:
    
        n_tracados = np.random.randint(0, len(array_tracados_maiores))
        tracado = array_tracados_maiores[n_tracados]
        
        vetor_medial_path = backgen.retorna_paths(f'{tracados_dir_maiores}/{tracado}')        
       
        resultados = backgen.inserir_vasos(vetor_medial_path[0],vetor_medial_path[1],vetor_pickles,pickle_dir,background_com_pad,treshold=30)       
        if resultados is not None:
            vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1 = resultados
            resultados_not_none += 1
           
            fundo_com_vasos = backgen.inserir_mapa(fundo_com_vasos,vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1, possui_mapas)
            contador +=1
        else:
            resultados_none += 1
  
    # plt.figure(figsize=[10, 8])
    # plt.title("fundo_com_vasos")
    # plt.imshow(fundo_com_vasos, 'gray', vmin=0, vmax=255)
    # plt.plot()
    fundo_recortado = fundo_com_vasos[200:1304,200:1576]

    img1 = Image.fromarray(fundo_recortado.astype(np.uint8))
    path = f'{trein_dir}/100_fundos_10_maps/{nome_background}_{j+43}_com_{n_vasos}.tiff'
    img = img1.save(path)
# print(f'resultados_none: {resultados_none}')