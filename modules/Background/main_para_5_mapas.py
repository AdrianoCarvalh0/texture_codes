from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Utils import functions

import background_generation as backgen

#root_dir = f"/home/adriano/projeto_mestrado/modules"
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
img_dir = f'{root_dir}/Imagens/vessel_data/images'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x'

trein_dir = f'{root_dir}/Training_validation'

#pickle_dir = f'{root_dir}/Vessel_Models_pickle'
pickle_dir_5 = f'{trein_dir}/Maps/5_maps_5_images'
#pickle_dir_10 = f'{trein_dir}/Mapas/10_mapas_de_10_imagens'
#pickle_dir_40 = f'{trein_dir}/Mapas/160_mapas_de_40_imagens'
#pickle_dir_50 = f'{trein_dir}/Mapas/200_mapas_de_50_imagens'

#background_dir = f'{root_dir}/Background/Mapas_gerados_artificialmente'
#background_dir_5 = f'{trein_dir}/Backgrounds/5_backgrounds'
#background_dir_10 = f'{trein_dir}/Backgrounds/10_backgrounds'
#background_dir_40 = f'{trein_dir}/Backgrounds/40_backgrounds'
background_dir_50 = f'{trein_dir}/Backgrounds/50_backgrounds'


tracados_dir = root_dir/"Artificial_Lines/bezier_traces"
tracados_dir_tests = root_dir/"Artificial_lines/bezier_traces_tests"

vetor_pickles = functions.read_directories(pickle_dir_5)
array_backgrounds = functions.read_directories(background_dir_50)
#array_tracados = funcoes.ler_diretorios(tracados_dir)
array_tracados_maiores = functions.read_directories(tracados_dir_tests)

problema = 0
resultados_not_none = 0
resultados_none = 0

#n_random = np.random.randint(0, len(vetor_pickles))  
#path_pickle = (pickle_dir_50 + f'/{vetor_pickles[n_random]}')
#print(path_pickle)

for j in range(100):
    
    imagem_binaria_sem_artefatos_laterais = None
    while imagem_binaria_sem_artefatos_laterais is None:
        n_random = np.random.randint(0, len(vetor_pickles))  
        path_pickle = (pickle_dir_5 + f'/{vetor_pickles[n_random]}')
        print(f'path_pickle: {path_pickle}')

        arquivo_pickle = pickle.load(open(path_pickle, 'rb')) 
        vessel_map = arquivo_pickle['vessel_model'].vessel_map 
        mapa_original = vessel_map.mapped_values

        mapa_original_norm = None
        imagem_binaria_original = vessel_map.mapped_mask_values
        imagem_binaria_sem_artefatos_laterais = backgen.returns_binary_image_without_artifacts(vessel_map, imagem_binaria_original)
    #imagem_binaria_sem_artefatos = backgen.fill_holes(imagem_binaria_sem_artefatos_laterais) 

    nro_norms_falhos = 0
    while mapa_original_norm is None:
        n_backgrounds = np.random.randint(0, len(array_backgrounds))
        background = np.array(Image.open(f'{background_dir_50}/{array_backgrounds[n_backgrounds]}'))        
        mapa_original_norm = backgen.normalize(background,mapa_original,imagem_binaria_original,30)
        nro_norms_falhos +=1
   
    nome_background = f'{array_backgrounds[n_backgrounds]}'
    background_recortado = background[0:1100,0:1370]
    nome_background = f'{array_backgrounds[n_backgrounds]}'
    nome_background = nome_background.replace("'","").replace(".tiff","")
    background_com_pad = np.pad(background_recortado, ((200,200),(200,200)), mode="symmetric", reflect_type="even")    
    background_bin = np.zeros(background_com_pad.shape)
    fundo_com_vasos2 = background_bin.copy()
    fundo_com_vasos = background_com_pad.copy()    
    possui_mapas =  np.full(shape = background_com_pad.shape, fill_value=0)
    possui_mapas2 =  np.full(shape = background_bin.shape, fill_value=0)
    n_vasos = np.random.randint(20, 50)    
    contador = 0
    while contador < n_vasos:
    
        n_tracados = np.random.randint(0, len(array_tracados_maiores))
        tracado = array_tracados_maiores[n_tracados]
        
        vetor_medial_path = backgen.return_paths(tracados_dir_tests/f"{tracado}")        
       
        resultados = backgen.insert_vessels(vetor_medial_path[0],vetor_medial_path[1],vetor_pickles,pickle_dir_5,background_com_pad,30)       
        if resultados is not None:
            vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1 = resultados
            resultados_not_none += 1
           
            fundo_com_vasos = backgen.insert_map(fundo_com_vasos,vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1, possui_mapas)           

            fundo_com_vasos2 = backgen.insert_binary_map(fundo_com_vasos2,vaso_sem_artefatos,possui_mapas2)
            contador +=1
        else:
            resultados_none += 1  
   
    fundo_recortado = fundo_com_vasos[200:1304,200:1576]
    fundo_recortado2 = fundo_com_vasos2[200:1304,200:1576]

    img1 = Image.fromarray(fundo_recortado.astype(np.uint8))
    path = f'{root_dir}/Images/Background_with_vessels_tests/images/{nome_background}_{j}_com_{n_vasos}.tiff'
    img = img1.save(path)

    img2 = Image.fromarray(fundo_recortado2.astype(np.bool_))
    path = f'{root_dir}/Images/Background_with_vessels_tests/labels/{nome_background}_{j}_com_{n_vasos}.tiff'
    img = img2.save(path)   
    print(f'laço: {j}')    
    print(f"número de falhas na normalização: {nro_norms_falhos}")


print(f'resultados_none: {resultados_none}')