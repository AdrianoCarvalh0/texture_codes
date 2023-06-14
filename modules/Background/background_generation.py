import sys
sys.path.insert(0, "C:\\Users\\adria\\Documents\\Mestrado\\texture_codes\\modules")

import json, tracemalloc, time
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from Slice_mapper import slice_mapper_util as smutil
from shapely.geometry import Point,LineString

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

    #Função com uma linha para inverter todos os valores
    # path1 = [np.array(item)[:,::-1] for item in q]
    return array_paths

def novos_pontos(caminho, escalar):
  novo_caminho = caminho.copy()
  novo_caminho[:, 1] = caminho[:, 1] + escalar
  return novo_caminho

def retorna_caminhos_transladados(source, increase=None):
  if increase == None:
    # retorna as menores e as maiores linhas e colunas dos caminhos
    # menor_linha, menor_coluna = np.min(medial_path, axis=1), np.min(medial_path, axis=0)
    min_coluna, min_linha = np.min(source[0], axis=0)

    # pega o primeiro_ponto na posição da menor coluna, e da menor linha, decrescidos do padding
    primeiro_ponto = np.array([min_coluna, min_linha])

    # # absorve os valores dos caminhos decrescidos do primmeiro ponto, varrendo todos os dois vetores
    caminhos_transl = [caminho - primeiro_ponto for caminho in source]

    return caminhos_transl

  else:
    res_factor = increase
    caminho_interp = smutil.increase_path_resolution(source, res_factor=res_factor)
   
    min_coluna, min_linha = np.min(caminho_interp[0], axis=0)

    # pega o primeiro_ponto na posição da menor coluna, e da menor linha, decrescidos do padding
    primeiro_ponto = np.array([min_coluna, min_linha])

    # # absorve os valores dos caminhos decrescidos do primmeiro ponto, varrendo todos os dois vetores
    caminhos_transl_interp = [caminho - primeiro_ponto for caminho in caminho_interp]

    return caminhos_transl_interp

def plotar_pontos(x,y, titulo=None):  
  idx = range(len(x))
  fig, ax = plt.subplots()
  plt.imshow(np.zeros((32,52)),'binary')
  ax.scatter(x, y)
  for i, txt in enumerate(idx):
    ax.annotate(txt, (x[i], y[i]))
  if titulo:
    plt.title(titulo)

def transform_v2(src, dst, img, order=0):
    """Transform image."""
    
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    
    if img.ndim==2:
        # Add channel dimension if 2D
        img = img[...,None]
    num_rows, num_cols, num_channels = img.shape
    
    # Find minimum and maximum values for dst points
    min_dst_col, min_dst_row = dst.min(axis=0).astype(int)
    max_dst_col, max_dst_row = np.ceil(dst.max(axis=0)).astype(int)
    ul_point_dst = np.array([min_dst_col, min_dst_row])
    # New origin point for the space containing both src and dst
    new_origin = np.minimum(ul_point_dst, np.zeros(2, dtype=int))
    translation = np.abs(new_origin)

    src -= new_origin
    dst -= new_origin

    # Create new source image considering the new origin
    img_proper = np.zeros((num_rows+translation[1], num_cols+translation[0], num_channels), dtype=img.dtype)
    img_proper[translation[1]:translation[1]+num_rows, translation[0]:translation[0]+num_cols] = img
    output_shape = (max_dst_row-new_origin[1], max_dst_col-new_origin[0])

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    img_out = warp(img_proper, tform.inverse, output_shape=output_shape, order=order)
    
    if img.ndim==2:
        img_out = img_out[0]
        
    return img_proper, img_out, src, dst


def plot(img_proper, img_out, src, dst, vmax):

    plt.figure(figsize=[10,5])
    plt.subplot(1, 2, 1)
    plt.imshow(img_proper, 'gray',  vmin=0, vmax=vmax)
    plt.plot(src[:,0], src[:,1], 'o')
    plt.title('Adjusted source image')
    plt.axis((0, img_proper.shape[1], img_proper.shape[0], 0))
    plt.subplot(1, 2, 2)
    plt.imshow(img_out, 'gray', vmin=0, vmax=vmax)
    plt.plot(dst[:,0], dst[:,1], 'o')
    plt.axis((0, img_out.shape[1], img_out.shape[0], 0))

def retorna_maior_linha(linha1,linha2,linha3):
  tam_linha1 = len(linha1.coords)
  tam_linha2 = len(linha2.coords)
  tam_linha3 = len(linha3.coords)
  x = np.array([tam_linha1,tam_linha2,tam_linha3])
  tam_max_linha = np.max(x)              
  return tam_max_linha

def retorna_novos_pontos_das_linhas(distance,linha):  
  points = []
  vetor = []
  for dist in distance:
    p = linha.interpolate(dist, normalized=True)
    points.append(p) 
  for p in points:
    vetor.append([p.x,p.y])  
  return vetor

def retornar_imagem_binaria_sem_artefatos(vessel_map, img_bin):

  num_rows, num_cols = img_bin.shape

  linha_minima = int(np.min(np.rint(vessel_map.path2_mapped))-1)
  linha_maxima  = int(np.max(np.rint(vessel_map.path1_mapped))+1)

  imagem_binaria_sem_artefatos = img_bin.copy().astype('int32')

  for num_row in range(int(linha_minima)):
    for num_col in range(num_cols):
      imagem_binaria_sem_artefatos[num_row,num_col] = 0 

  for i in range(linha_maxima, num_rows):
    for num_col in range(num_cols):
      imagem_binaria_sem_artefatos[i,num_col] = 0
  return imagem_binaria_sem_artefatos


#código gerado pelo Matheus
def estimate_background(image: np.ndarray, label: np.ndarray, window_size: int=15) -> np.ndarray:

    contains_foreground_pixels = lambda target_label: np.count_nonzero(target_label) > 0

    # divide a imagem em patches
    list_slices = []
    window_size = 15
    h_window_size = window_size // 2
    i = h_window_size
    j = h_window_size

    while i < image.shape[0] - h_window_size:
        j = h_window_size
        while j < image.shape[1] - h_window_size:
            list_slices.append((slice(i-h_window_size, i+h_window_size+1), slice(j-h_window_size, j+h_window_size+1)))
            j += h_window_size
        i += h_window_size
    
    # separa os patches de background e foreground
    only_background = np.zeros_like(image)
    only_foreground = np.zeros_like(image)
    foreground_patches = []
    background_patches = []
    foreground_centers = []
    background_centers = []

    for sl in list_slices:
        center = (sl[0].start + h_window_size, sl[1].start + h_window_size)
        if contains_foreground_pixels(label[sl[0], sl[1]]):
            foreground_patches.append(sl)
            foreground_centers.append(center)
        else:
            background_patches.append(sl)
            background_centers.append(center)

    foreground_dm = distance_matrix(foreground_centers, background_centers)

    for fp in foreground_patches:
        only_foreground[fp[0], fp[1]] = image[fp[0], fp[1]]

    for bp in background_patches:
        only_background[bp[0], bp[1]] = image[bp[0], bp[1]]


    # substitui os patches de background e foreground
    background_patches = np.array(background_patches)
    foreground_patches = np.array(foreground_patches)
    n_closest_patches = 10

    generated_background = only_background.copy()

    for idx, fp in enumerate(foreground_patches):
        # patches de background que são mais próximos a cada patch de foreground
        closest_ind = np.argsort(foreground_dm[idx])
        closest_background_patches = background_patches[closest_ind][:n_closest_patches]

        # substitui o patch de foreground com um patch de background aleatório
        random_idx = np.random.randint(0, len(closest_background_patches))
        sl = closest_background_patches[random_idx]
        generated_background[fp[0], fp[1]] = image[sl[0], sl[1]]
    
    return generated_background

def retorna_linhas_offset_posicao_tamanho(mapa,caminhos):
  #Algoritmo usando LineString e OffsetCurve
  mapa = np.array(mapa)
  rows, cols = mapa.shape[0], mapa.shape[1]
  distancia = (rows/2)
  linha_c  = LineString(caminhos)
  #import pdb; pdb.set_trace()
  linha_offset_esq = linha_c.offset_curve(distance=-distancia,  join_style=1)
  linha_offset_dir = linha_c.offset_curve(distance=distancia, join_style=1)

  maior_tamanho = retorna_maior_linha(linha_offset_esq, linha_c, linha_offset_dir)

  return linha_c,linha_offset_esq,linha_offset_dir, maior_tamanho

def retorna_dst_array_np(linha_centro,linha_esquerda,linha_direita,maior_tam):
  distance = np.linspace(0,1,maior_tam)
  dst_array = []
  vetor_linha_esquerda = retorna_novos_pontos_das_linhas(distance,linha_esquerda)
  vetor_linha_central = retorna_novos_pontos_das_linhas(distance,linha_centro)
  vetor_linha_direita = retorna_novos_pontos_das_linhas(distance,linha_direita)

  for l_e in vetor_linha_esquerda:
    dst_array.append(l_e)
  for l_c in vetor_linha_central:
    dst_array.append(l_c)
  for l_d in vetor_linha_direita:
    dst_array.append(l_d)
  dst_arr_np = np.array(dst_array)
  return dst_arr_np