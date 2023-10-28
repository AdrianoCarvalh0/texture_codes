from pathlib import Path
import numpy as np
from PIL import Image
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Funcoes_gerais import funcoes

import background_generation as backgen

#root_dir = f"/home/adriano/projeto_mestrado/modules"
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
img_dir = f'{root_dir}/Imagens/vessel_data/images'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x'

trein_dir = f'{root_dir}/Treinamento_validacao'

#pickle_dir = f'{root_dir}/Vessel_Models_pickle'
#pickle_dir_5 = f'{trein_dir}/Mapas/5_mapas_de_5_imagens'
#pickle_dir_10 = f'{trein_dir}/Mapas/10_mapas_de_10_imagens'
#pickle_dir_40 = f'{trein_dir}/Mapas/160_mapas_de_40_imagens'
pickle_dir_50 = f'{trein_dir}/Mapas/200_mapas_de_50_imagens'

#background_dir = f'{root_dir}/Background/Mapas_gerados_artificialmente'
#background_dir_5 = f'{trein_dir}/Backgrounds/5_backgrounds'
#background_dir_10 = f'{trein_dir}/Backgrounds/10_backgrounds'
#background_dir_40 = f'{trein_dir}/Backgrounds/40_backgrounds'
background_dir_50 = f'{trein_dir}/Backgrounds/50_backgrounds'


tracados_dir = root_dir/"Artificial_Lines/tracados_bezier"
tracados_dir_maiores = root_dir/"Artificial_Lines/tracados_bezier_maiores"

vetor_pickles = funcoes.ler_diretorios(pickle_dir_50)
array_backgrounds = funcoes.ler_diretorios(background_dir_50)
#array_tracados = funcoes.ler_diretorios(tracados_dir)
array_tracados_maiores = funcoes.ler_diretorios(tracados_dir_maiores)

problema = 0
resultados_not_none = 0
resultados_none = 0

n_random = np.random.randint(0, len(vetor_pickles))  
path_pickle = (pickle_dir_50 + f'/{vetor_pickles[n_random]}')
print(path_pickle)

for j in range(100):
    n_random = np.random.randint(0, len(vetor_pickles))  
    path_pickle = (pickle_dir_50 + f'/{vetor_pickles[n_random]}')
    print(path_pickle)
    n_backgrounds = np.random.randint(0, len(array_backgrounds))
    background = np.array(Image.open(f'{background_dir_50}/{array_backgrounds[n_backgrounds]}'))
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
        
        vetor_medial_path = backgen.retorna_paths(tracados_dir_maiores/f"{tracado}")        
       
        resultados = backgen.inserir_vasos(vetor_medial_path[0],vetor_medial_path[1],vetor_pickles,pickle_dir_50,background_com_pad,treshold=30,path_pickle=path_pickle)       
        if resultados is not None:
            vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1 = resultados
            resultados_not_none += 1
           
            fundo_com_vasos = backgen.inserir_mapa(fundo_com_vasos,vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1, possui_mapas)           

            fundo_com_vasos2 = backgen.inserir_mapa_bin(fundo_com_vasos2,vaso_sem_artefatos,possui_mapas2)
            contador +=1
        else:
            resultados_none += 1
  
    # plt.figure(figsize=[10, 8])
    # plt.title("fundo_com_vasos")
    # plt.imshow(fundo_com_vasos, 'gray', vmin=0, vmax=255)
    # plt.plot()
    fundo_recortado = fundo_com_vasos[200:1304,200:1576]
    fundo_recortado2 = fundo_com_vasos2[200:1304,200:1576]

    img1 = Image.fromarray(fundo_recortado.astype(np.uint8))
    path = f'{trein_dir}/Imagens_Artificiais/Geradas_a_partir_de_1_mapa/pack1/imagens_artificiais/{nome_background}_{j}_com_{n_vasos}.tiff'
    img = img1.save(path)

    img2 = Image.fromarray(fundo_recortado2.astype(np.bool_))
    path = f'{trein_dir}/Imagens_Artificiais/Geradas_a_partir_de_1_mapa/pack1/labels/{nome_background}_{j}_com_{n_vasos}.tiff'
    img = img2.save(path)

    #plt.figure(figsize=[10, 8])
    #plt.title("img2")
    #plt.imshow(img2, 'gray', vmin=0, vmax=1)


# print(f'resultados_none: {resultados_none}')