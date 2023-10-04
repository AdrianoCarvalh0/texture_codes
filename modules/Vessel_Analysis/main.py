import sys
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")


import numpy as np
import pickle

from PIL import Image
# This is a sample Python script.

import vessel_analysis as va

if __name__ == '__main__':   
  
  imag = 'T-3 Weeks@Females@926 F@926-CTL-middle-20X-02'

  #imag = '3D P0@CTL-3-FC-A'

  pasta_mestrado ="/home/adriano/projeto_mestrado/modules" 
 
  arquivo = f"{pasta_mestrado}/Vetores_Extraidos_json/novos/{imag}.json"
  
  caminho_img = f"{pasta_mestrado}/Imagens/vessel_data/images/{imag}.tiff"

  #pega o arquivo e armazena em um array
  array_path = va.retorna_paths(arquivo)

  #leitura da imagem
  img = np.array(Image.open(caminho_img))

  #pega a metade inteira do vetor
  half_array = len(array_path)//2

  x=0
  for i in range(half_array):  
    img, caminhos_transladados, primeiro_ponto = va.redimensiona_imagem(array_path[x:x+2], caminho_img)     
    alcance = va.setar_alcance(array_path[0], array_path[1])
    vessel_mod,  cross_t = va.gera_vessel_cross(img, caminhos_transladados[0], caminhos_transladados[1], alcance)   
    #va.plot_figure(img, vessel_mod, cross_t)
    #plot_figure2(img, vessel_mod, cross_t)
    
    #parte para salvar o .pickle
    data_dump = {"img_file": caminho_img, "vessel_model": vessel_mod, "primeiro_ponto": primeiro_ponto} 
    savedata = f'{pasta_mestrado}/Vessel_Models_pickle/novos/{imag}_savedata{i}.pickle'
    pickle.dump(data_dump, open(savedata,"wb"))  
    x+=2

