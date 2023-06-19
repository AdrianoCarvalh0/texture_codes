import sys
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

import numpy as np
import matplotlib.pyplot as plt
import json
from Slice_mapper import slice_mapper as slice_mapper
from PIL import Image

# lê um arquivo e retorna um array dos dados do arquivo
# O array está no formato coluna/linha. O código que faz a extração grava no formato coluna/linha
# Tem várias partes do vessel analysis e do slice_mapper que usam desta forma.
def retorna_paths(arq_json):
    """Função que lê um arquivo json retorna os paths 1 e 2 de uma ou várias marcações manuais dos vasos sanguíneos

    Parâmetros:
    -----------
    arq_json: str
        arquivo que contém as coordenadas, linhas e colunas da localização do vaso sanguíneo com extensão .json
    Retorno:
    -----------
    array_paths: list, contendo ndarray
        retorna path1 e path2 de um ou vários vasos extraídos.
        Os valores armazenados no path1 e path2 são as demarcações manuais feitas nos vasos.
    """
    # leitura do json
    q = json.load(open(arq_json, 'r'))

    # transforma todos os itens lidos em np.array
    array_paths = [np.array(item) for item in q]

    # Função com uma linha para inverter todos os valores
    # path1 = [np.array(item)[:,::-1] for item in q]
    return array_paths


def setar_alcance(array_1, array_2, alcance_extra=6):
    """ Função que pega os dois primeiros valores das linhas e colunas dos vetores passados por parâmetro e
        retorna o alcance. Calcula a distância Euclidiana entre os quatro pontos.

    Parâmetros:
    -----------
    array_1: ndarray, float
        vetor do caminho 1
    array_2: ndarray, float
        vetor do caminho 2
    Retorno:
    -----------
    alcance: int
        Retorna o valor inteiro do cálculo da distância Euclidiana.
        Serve para delimitar a região que será exibida da imagem.
    """
    # pega a coluna 1 do vetor 1
    coluna1 = array_1[0][0]

    # pega a linha 1 do vetor 1
    linha1 = array_1[0][1]

    # pega a coluna 2 do vetor 2
    coluna2 = array_2[0][0]

    # pega a linha 2 do vetor 2
    linha2 = array_2[0][1]

    # O alcance vai ser a raiz quadrada do resultado da diferença quadrática entre os dois pontos dos dois vetores - distância Euclidiana
    # A variável alcance_extra permite que a região setada pelo alcance seja um pouco maior. O alcance é aumentado, tanto acima, como abaixo dos valores mapeados
    alcance = int(np.sqrt((linha1 - linha2) ** 2 + (coluna1 - coluna2) ** 2) + alcance_extra)

    return alcance


def retorna_linhas_colunas(caminhos):
    """ Função que retorna os valores mínimos e máximos de dois vetores

    Parâmetros:
    -----------
    caminhos: list, contendo ndarray
        lista que contém dois vetores ndarray (path1 e path2)
    Retorno:
    -----------
    min_linha: int
        valor mínimo encontrado entre as linhas
    min_coluna: int
        valor mínimo encontrado entre as colunas
    max_linha: int
        valor máximo encontrado entre as linhas
    max_coluna: int
        valor máximo encontrado entre as colunas
    """
    # pega a primeira posição do vetor
    caminho1 = caminhos[0]

    # pega a segunda posição do vetor
    caminho2 = caminhos[1]

    min_coluna1, min_linha1 = np.min(caminho1, axis=0)
    min_coluna2, min_linha2 = np.min(caminho2, axis=0)

    max_coluna1, max_linha1 = np.max(caminho1, axis=0)
    max_coluna2, max_linha2 = np.max(caminho2, axis=0)

    min_coluna = int(np.min([min_coluna1, min_coluna2]))
    min_linha = int(np.min([min_linha1, min_linha2]))
    max_coluna = int(np.max([max_coluna1, max_coluna2]))
    max_linha = int(np.max([max_linha1, max_linha2]))

    return min_linha, min_coluna, max_linha, max_coluna


def redimensiona_imagem(caminhos, caminho_da_img):
    """ Função que pega um vetor, contendo os paths 1, 2 e o endereço de uma imagem, recriando sua dimensão para
     as setadas pelas variáveis que contém os maiores e menores valores das linhas e colunas contidas nos paths 1 e 2

    Parâmetros:
    -----------
    caminhos: list, contendo ndarray
        lista que contém dois vetores ndarray (path1 e path2)
    caminho_da_img: str
        endereço da imagem que exibe a sua loclização
    Retorno:
    -----------
    img1: ndarray, image
        imagem redimensionada
    caminhos_transladados: list, float
        lista contendo um par de vetores que foram transladados a partir do primeiro ponto da menor linha
        e coluna, menos um padding, para encaixar na imagem redimensionada
    primeiro_ponto: ndarray
        contém as informações da menor linha e coluna, menos um padding da imagem original
    """
    # padding setado para mostrar uma região um pouco maior do que a do vaso em questão
    padding = 5

    # retorna as menores e as maiores linhas e colunas dos caminhos
    menor_linha, menor_coluna, maior_linha, maior_coluna = retorna_linhas_colunas(caminhos)

    # pega o primeiro_ponto na posição da menor coluna, e da menor linha, decrescidos do padding
    primeiro_ponto = np.array([menor_coluna - padding, menor_linha - padding])

    # absorve os valores dos caminhos decrescidos do primmeiro ponto, varrendo todos os dois vetores
    caminhos_transladados = [caminho - primeiro_ponto for caminho in caminhos]

    # imagem que absorve a img_path e mostra a região delimitada pelos parâmetros pelas menores e maiores linhas/colunas
    # o parâmetro "-padding" permite que seja pega uma região um pouco maior em todos os valores das linhas/colunas
    img1 = np.array(Image.open(caminho_da_img))[menor_linha - padding:maior_linha + padding,
           menor_coluna - padding:maior_coluna + padding]

    return img1, caminhos_transladados, primeiro_ponto


def gera_vessel_cross(img, caminhos_trans0, caminhos_trans1, alcance, delta_eval=1.,smoothing=0.01):
    """ Função que cria o modelo de vaso e os caminhos transversais
   
    Parâmetros:
    -----------
    img: ndarray, float
       imagem redimensionada contendo a área do vaso extraído
    caminhos_trans0: ndarray, float
        caminhos transladados na posição 0 do vetor de caminhos transladados
    caminhos_trans1: ndarray, float
        caminhos transladados na posição 1 do vetor de caminhos transladados
    alcance: int
        variável que define o quanto de limite superior e inferior a imagem terá, tem implicação direta com a quantidade de linhas do mapa criado
    delta_eval: float
        variável que define os intervalos
    smoothing: float
        variável que define o grau da suavização
    Retorno:
    -----------
    vessel_model: obejct VesselModel
        retorna o modelo do vaso com um objeto instanciado da classe VesselModel
    cross_paths: ndarray, float
        caminhos transversais
    """
    vessel_model, cross_paths = slice_mapper.map_slices(img, caminhos_trans0, caminhos_trans1, delta_eval, smoothing, alcance)

    return vessel_model, cross_paths
def plot_figure(img, vessel_model, cross_paths):       
    """ Função que cria o modelo de vaso e os caminhos transversais

    Parâmetros:
    -----------
    img: ndarray, float
       imagem redimensionada contendo a área do vaso extraído
    vessel_model: obejct VesselModel
        retorna o modelo do vaso com um objeto instanciado da classe VesselModel
    cross_paths: ndarray, float
        caminhos transversais
    Retorno:
    -----------
        plota a imagem redimensionada, juntamente com o modelo do vaso, os caminhos tranversais, os paths 1 e 2
        transladados, de três maneiras diferentes:
        1 - com os valores mapeados tendo o mínimo em 0 e máximo em 60
        2 - valores mapeados no padrão, de 0 a 255
        3 - valores mapeados entre o mínimo 0 e máximo nos valores encontrados no mapeamento
    """

    vessel_map = vessel_model.vessel_map
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot()
    slice_mapper.plot_model(img, vessel_model, cross_paths, ax)   
    #plt.title("Imagem analisada")
    norm = ax.images[0].norm
    norm.vmin, norm.vmax = 0, 60
    plt.axis('off')
    #arquivo_modelo = f'{pasta_mestrado}/Imagens/mapas/modelo/{imag}_{x}.png'
    #plt.imsave(arquivo_modelo, vessel_map.mapped_values, cmap='gray', vmin=0, vmax=60)  
     
    
    plt.figure(figsize=[12, 10])
    #plt.title("Vmin=0 e Vmax=60")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)
    #plt.plot(vessel_map.path1_mapped, c='green')
    #plt.plot(vessel_map.path2_mapped, c='green')
    plt.axis('off') 
    #plt.imsave(imag+'.png', vessel_map.mapped_values, cmap='gray', vmin=0, vmax=60)
    #arquivo_min0max60 = f'{pasta_mestrado}/Imagens/mapas/min0max60/{imag}_{x}.png'
    #plt.imsave(arquivo_min0max60, vessel_map.mapped_values, cmap='gray', vmin=0, vmax=60)     
 
    plt.figure(figsize=[12, 10])
    plt.title("Vmin=0 e Vmax=255")
    plt.plot()
    plt.imshow(vessel_map.mapped_values[::-1], 'gray', vmin=0, vmax=255)
    #plt.plot(vessel_map.path1_mapped, c='green')
    #plt.plot(vessel_map.path2_mapped, c='green')
    plt.axis('off')
    #arquivo_min0max255 = f'{pasta_mestrado}/Imagens/mapas/min0max255/{imag}_{x}.tiff'
    #plt.imsave(arquivo_min0max255, vessel_map.mapped_values, cmap='gray', vmin=0, vmax=255)
    #image = np.array(vessel_map.mapped_values, dtype=np.uint8)
    #imsave(arquivo_min0max255, image)  

    plt.figure(figsize=[12, 10])
    plt.title("Vmin=0 e Vmax=maximo valor mapeado")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=vessel_map.mapped_values.max())     
    #plt.plot(vessel_map.path1_mapped, c='green')
    #plt.plot(vessel_map.path2_mapped, c='green')
    plt.axis('off')
    #arquivo_min0max = f'{pasta_mestrado}/Imagens/mapas/min0maxmapeado/{imag}_{x}.png'
    #plt.imsave(arquivo_min0max, vessel_map.mapped_values, cmap='gray', vmin=0, vmax=vessel_map.mapped_values.max())
    
def plot_figure2(img, vessel_model, cross_paths):       
    """ Função que cria o modelo de vaso e os caminhos transversais

    Parâmetros:
    -----------
    img: ndarray, float
       imagem redimensionada contendo a área do vaso extraído
    vessel_model: obejct VesselModel
        retorna o modelo do vaso com um objeto instanciado da classe VesselModel
    cross_paths: ndarray, float
        caminhos transversais
    Retorno:
    -----------
        plota a imagem redimensionada, juntamente com o modelo do vaso, os caminhos tranversais, os paths 1 e 2
        transladados, de três maneiras diferentes:
        1 - com os valores mapeados tendo o mínimo em 0 e máximo em 60
        2 - valores mapeados no padrão, de 0 a 255
        3 - valores mapeados entre o mínimo 0 e máximo nos valores encontrados no mapeamento
    """

    vessel_map = vessel_model.vessel_map
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot()
    slice_mapper.plot_model(img, vessel_model, cross_paths, ax)        
    norm = ax.images[0].norm
    norm.vmin, norm.vmax = 0, 60
    #arquivo_modelo = f'{pasta_mestrado}/Imagens/plots/modelo/{imag}_{x}.png'
    #plt.savefig(arquivo_modelo)  
    
    plt.figure(figsize=[12, 10])    
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')
    #arquivo_min0max60 = f'{pasta_mestrado}/Imagens/plots/min0max60/{imag}_{x}.png'
    #plt.savefig(arquivo_min0max60)

    plt.figure(figsize=[12, 10])    
    plt.plot()
    plt.imshow(vessel_map.mapped_values[::-1], 'gray', vmin=0, vmax=255)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')    
    #arquivo_min0max255 = f'{pasta_mestrado}/Imagens/plots/min0max255/{imag}_{x}.png'
    #plt.savefig(arquivo_min0max255)

    plt.figure(figsize=[12, 10])   
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=vessel_map.mapped_values.max())     
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')    
    #arquivo_min0max = f'{pasta_mestrado}/Imagens/plots/min0maxmapeado/{imag}_{x}.png'
    #plt.savefig(arquivo_min0max)
    