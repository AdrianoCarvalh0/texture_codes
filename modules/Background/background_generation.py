import sys
import scipy
import pickle
import skimage as ski
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/Slice_mapper")

import json, tracemalloc, time
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
import slice_mapper_util as smutil
from shapely.geometry import Point,LineString
from PIL import Image
from scipy import ndimage

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

def encontrar_pixel_mais_frequente(mapa):

  image = Image.fromarray(mapa)

  # Converte a imagem para escala de cinza
  image_gray = image.convert("L")

  # Obtém o histograma dos valores de pixel
  histogram = image_gray.histogram()

  # Cria uma lista de tuplas (valor de pixel, frequência)
  pixel_freq_pairs = list(enumerate(histogram))

  # Ordena a lista em ordem decrescente de frequência
  sorted_pixel_freq_pairs = sorted(pixel_freq_pairs, key=lambda x: x[1], reverse=True)

  # Separa os valores de pixel e as frequências
  pixels, freqs = zip(*sorted_pixel_freq_pairs)

  # Encontra o valor do pixel com a maior frequência
  pixel_mais_frequente = pixels[0]

  return pixel_mais_frequente


def encontrar_mediana_fundo_mapa(img,img_label):
    
    ints_fundo_mapa = img[img_label==0]
    threshold = np.median(ints_fundo_mapa)

    return threshold
   

def novos_pontos(caminho, escalar):
  novo_caminho = caminho.copy()
  novo_caminho[:, 1] = caminho[:, 1] + escalar
  return novo_caminho

def retorna_caminhos_transladados(source, increase=None):
  if increase == None:
    # retorna as menores e as maiores linhas e colunas dos caminhos
    # menor_linha, menor_coluna = np.min(medial_path, axis=1), np.min(medial_path, axis=0)
    min_coluna, min_linha = np.min(source, axis=0)

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
        
    return img_proper, img_out, src, dst, tform, translation, new_origin


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


def mask_binary_vessel(img, vessel_map):
   
  linha_minima = int(np.min(np.rint(vessel_map.path2_mapped)))
  linha_maxima  = int(np.max(np.rint(vessel_map.path1_mapped)))
  mask_bin_vessel = np.ones(img.shape).astype('int32')


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

def retorna_linhas_offset_posicao_tamanho(caminhos,distancia):
  #Algoritmo usando LineString e OffsetCurve
  
  linha_c  = LineString(caminhos)
 
  linha_offset_esq = linha_c.offset_curve(distance=-distancia,  join_style=1)
  linha_offset_dir = linha_c.offset_curve(distance=distancia, join_style=1)

  maior_tamanho = retorna_maior_linha(linha_offset_esq, linha_c, linha_offset_dir)

  return linha_offset_esq, linha_c, linha_offset_dir, maior_tamanho

def retorna_dst_array_np(linha_esquerda,linha_centro,linha_direita,maior_tam):
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


def expandir_mapas_do_tamanho_do_tracado(mapa_original,maior_valor):
  #import pdb; pdb.set_trace()
  rows, cols = mapa_original.shape
  fator = maior_valor/cols
  fator_int = int(fator)  
  resto = fator - fator_int
  mult = int(resto*cols)
  
  imagem_replicada = np.tile(mapa_original, (1,fator_int))
  rows_rep, cols_rep = imagem_replicada.shape

  maior_valor_int = (cols_rep+mult)

  imagem_replicada_total = np.zeros((rows, maior_valor_int))
  
  imagem_replicada_total[0:rows_rep,0:cols_rep] = imagem_replicada
  imagem_replicada_total[0:rows_rep,cols_rep:maior_valor_int] = mapa_original[0:rows,0:mult]
  
  return imagem_replicada_total

  
 
def inserindo_vaso_no_fundo(img,img_label,backg,point):

  # img_out_sq = img.squeeze()
  # img_out_transf_sq = img_label.squeeze()

  non_zero = np.nonzero(img_label)
  non_zero_t = np.transpose(non_zero)
  
  vetor_rows_back = []
  vetor_cols_back = []
  vetor_rows = []
  vetor_cols = []

  for i in range(len(non_zero_t)):
    rows = non_zero_t[i][0]
    cols = non_zero_t[i][1]
    rows_back = rows + point[0]
    vetor_rows.append(rows)
    vetor_rows_back.append(rows_back)

    cols_back = cols+point[1]
    vetor_cols.append(cols)
    vetor_cols_back.append(cols_back)
  
  for i in range(len(vetor_rows_back)):
    backg[vetor_rows_back[i],vetor_cols_back[i]] = img[vetor_rows[i],vetor_cols[i]]
  
  return backg

def delaunay_plot(img, img_out, tri, tri_inv):

    plt.figure(figsize=[12,6])
    ax = plt.subplot(121)
    plt.imshow(img)
    x, y = tri.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri.simplices.copy())

    ax = plt.subplot(122)
    plt.imshow(img_out)
    x, y = tri_inv.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri_inv.simplices.copy())


def inserindo_vaso_fundo2(possui_vasos, img,img_label,background,point,limiar):    
    
    numero = 1.8*10e100
    merged = np.full(shape = background.shape, fill_value=numero)
    img_out_bin_large = np.full(shape = background.shape, fill_value=0)

    img_out_large = np.full(shape = background.shape, fill_value=0)
    rows_img_out_sq, cols_img_out_sq = img.shape    

    merged[point[0]:(point[0]+rows_img_out_sq),point[0]:(point[0]+cols_img_out_sq)]=img
    img_out_bin_large[point[0]:(point[0]+rows_img_out_sq),point[0]:(point[0]+cols_img_out_sq)]=img_label
    img_out_large[point[0]:(point[0]+rows_img_out_sq),point[0]:(point[0]+cols_img_out_sq)]=img       
    limiar_mask = (merged <= limiar) #& (possui_vasos ==0)
    merged[limiar_mask] = background[limiar_mask] 
    merged[merged==numero] = background[merged==numero] 
    merged[img_out_bin_large==1]=img_out_large[img_out_bin_large==1]

    return merged,img_out_bin_large

def transf_map_dist(img_map,img_map_binario,img_vaso_bin,img_fundo):
  
  img_copy = img_map.copy()
  rows, cols = img_copy.shape
  img_vaso_bin_sq = img_vaso_bin.squeeze()
  img_dist = ndimage.distance_transform_edt(img_map_binario)
  img_dist[img_vaso_bin_sq] = 0
  img_probs = img_dist/img_dist.max()
  img_probs[img_vaso_bin_sq] = 2
  img_probs[img_map_binario==0] = 2
  img_rand = np.random.rand(img_map_binario.shape[0],img_map_binario.shape[1] )  
  inds = np.nonzero(img_rand>img_probs)  
  img_copy[inds] = img_fundo[0:rows,0:cols][inds]

  img_copy[img_vaso_bin_sq==1]=img_map[img_vaso_bin_sq==1]

  return img_copy 
  

def fill_holes(img_map_binario):
  img_map_bin_inv = 1 - img_map_binario

  s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
  img_label, num_comp = scipy.ndimage.label(img_map_bin_inv, s)
  tam_comp = scipy.ndimage.sum_labels(img_map_bin_inv, img_label, range(1, num_comp+1))
  inds = np.argsort(tam_comp)

  for idx in inds[:-2]:
      img_map_binario[img_label==idx+1] = 1

  return img_map_binario


def rotacionando_mapa_expandido(map, dst, max_tam):

  map_original_expanded = np.array(map)
  
  rows, cols = map_original_expanded.shape[0], map_original_expanded.shape[1]
  src_cols = np.linspace(0, cols, max_tam)
  src_rows = np.linspace(0, rows, 3)
  src_cols, src_rows = np.meshgrid(src_cols,src_rows)
  src = np.dstack([src_cols.flat, src_rows.flat])[0]

  img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = transform_v2(src, dst, map_original_expanded)

  return img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin

def criar_mascara_binaria_mapa(dst,img):
  
  tam = len(dst)
  div = tam//3
  vetor_div =[div,div*2,div*3]
  parte_direita = dst[0:vetor_div[0]]
  parte_esquerda = dst[vetor_div[1]:vetor_div[2]]

  lista_somada = np.concatenate((parte_esquerda, parte_direita[::-1]))
  polygon = np.array(lista_somada)[:,::-1]
  mask_map = ski.draw.polygon2mask(img.shape, polygon)
  mask_map.astype(int)
  mask_map_sq = mask_map.squeeze()

  return mask_map_sq

def criar_mascara_binaria_vaso(ves_map, new_orig,caminhos,img):
  linha_minima = int(np.min(np.rint(ves_map.path2_mapped)))
  linha_maxima  = int(np.max(np.rint(ves_map.path1_mapped)))

  diametro_maximo_vaso = (linha_maxima-linha_minima)/2

  linha_mask_central,linha_mask_esquerda,linha_mask_direita, max_tam = retorna_linhas_offset_posicao_tamanho(caminhos,diametro_maximo_vaso)

  dst_mask_np = retorna_dst_array_np(linha_mask_central,linha_mask_esquerda,linha_mask_direita, max_tam)
 
  dst_mask_np -= new_orig

  tam = len(dst_mask_np)

  div = tam//3

  vetor_div =[div,div*2,div*3]

  parte_direita = dst_mask_np[0:vetor_div[0]]  
  parte_esquerda = dst_mask_np[vetor_div[1]:vetor_div[2]] 

  lista_somada = np.concatenate((parte_esquerda, parte_direita[::-1]))
  polygon = np.array(lista_somada)[:,::-1]
  mask_vessel = ski.draw.polygon2mask(img.shape, polygon) # type: ignore
  mask_vessel.astype(int)
  mask_vessel_sq = mask_vessel.squeeze()

  return mask_vessel_sq

def criar_vaso_binario_expandido(mapa_bin,dst,max_tam):
  rows_bin, cols_bin = mapa_bin.shape[0], mapa_bin.shape[1]

  src_cols_bin = np.linspace(0, cols_bin, max_tam)
  src_rows_bin = np.linspace(0, rows_bin, 3)
  src_cols_bin, src_rows_bin = np.meshgrid(src_cols_bin,src_rows_bin)
  src_bin = np.dstack([src_cols_bin.flat, src_rows_bin.flat])[0]

  _, img_out_bin,_, _, _, _, _ = transform_v2(src_bin, dst, mapa_bin)

  return img_out_bin

def retirar_artefatos(img,mask_mapa):
  img_out_sq = img.squeeze()

  img_sem_artefatos = np.zeros(img_out_sq.shape,dtype=np.uint8)

  for i in range(img_sem_artefatos.shape[0]):
      for j in range(img_sem_artefatos.shape[1]):
          if mask_mapa[i, j] == True:  # Verifica se o pixel é branco na imagem booleana
              # Inserindo o pixel quando a máscara tem True
              img_sem_artefatos[i, j] = img_out_sq[i,j]

  return img_sem_artefatos

def normaliza(img_fundo,img_mapa,mask_vaso):
  
  ints_fundo = img_fundo.flatten()
  ints_mapa = img_mapa[mask_vaso==0]
  media_fundo = np.mean(ints_fundo)
  std_fundo = np.std(ints_fundo)
  media_mapa = np.mean(ints_mapa)
  std_mapa = np.std(ints_mapa)

  img_mapa_norm1 = (img_mapa - media_mapa)/std_mapa
  img_mapa_norm = img_mapa_norm1*std_fundo + media_fundo

  return img_mapa_norm


def merge(img_fundo,img_mapa,mask_vaso,p):
   img_map_copy = img_mapa.copy()
   pixeis =  np.nonzero(mask_vaso==0)
   num_pix = int(len(pixeis)*p)
   inds = np.random.choice(range(len(pixeis)), size=num_pix, replace=False)
   pixeis_replace_x = pixeis[0][inds]
   pixeis_replace_y = pixeis[1][inds]

   img_map_copy[pixeis_replace_x,pixeis_replace_y] = img_fundo[pixeis_replace_x,pixeis_replace_y]

   return img_map_copy


def histograma_matching(img_map,img_label_vaso, img_fundo):
    img_label_vaso_sq = img_label_vaso.squeeze()
    mapa_copia = img_map.copy()
    pos_vaso = (img_label_vaso_sq==1)
    mapa_copia[pos_vaso] = 0
    histograma_alt = ski.exposure.match_histograms(mapa_copia, img_fundo)
    histograma_alt[pos_vaso] = img_map[pos_vaso]

    return histograma_alt




# def inserir_mapa_no_fundo(fundo, mapa, mapa_bin, vaso_bin,limiar, possui_mapas, merged):
  
#    img_back = mapa.copy()
   
#    limiar_mask = (img_back <= limiar) & (mapa_bin == 1) & (vaso_bin == 0)
#    img_back[limiar_mask] = fundo

#    pix_map = np.nonzero(mapa_bin)
#    possui_mapas[pix_map] += 1 
#    merged[pix_map] += img_back[pix_map]

#    return merged, possui_mapas
 
 
def inserir_mapa(background,img_vaso_bin,img_mapa,img_mapa_bin, limiar, possui_mapas):
  merged_map = background.copy()#.astype('float64')
  img_mapa_copy = img_mapa.copy()     
  rows, cols = img_mapa.shape

  limiar_mask = (img_mapa <= limiar) & (img_mapa_bin == 1) & (img_vaso_bin == 0)

  img_mapa_copy[limiar_mask] = background[0:rows,0:cols][limiar_mask]

  pix_map = np.nonzero(img_mapa_bin & (possui_mapas[0:rows,0:cols] == 0)) 
  pix_vaso = np.nonzero(img_vaso_bin)
  possui_mapas[pix_vaso] += 1 
  merged_map[pix_map] =  img_mapa_copy[pix_map]
  
  return merged_map


def inserir_vasos(array_medial_path, distance, array_pickles,pickle_dir,back_artif):   
  n_random = np.random.randint(0, len(array_pickles))  
  
  path = (pickle_dir + f'/{array_pickles[n_random]}')
  print(f'{array_pickles[n_random]}')

  arquivo_pickle = pickle.load(open(path, 'rb')) 
  vessel_map = arquivo_pickle['vessel_model'].vessel_map 
  mapa_original = vessel_map.mapped_values
  
  imagem_binaria_original = vessel_map.mapped_mask_values 
  imagem_binaria_sem_artefatos_laterais = retornar_imagem_binaria_sem_artefatos(vessel_map, imagem_binaria_original)
  
  imagem_binaria_sem_artefatos = fill_holes(imagem_binaria_sem_artefatos_laterais) 
  
  mapa_original_norm = normaliza(back_artif,mapa_original,imagem_binaria_sem_artefatos)
  
  rows, cols = mapa_original.shape  

  distancia = (rows/2) 
  
  limiar1 = encontrar_pixel_mais_frequente(mapa_original_norm)  
  
  limiar2 = encontrar_mediana_fundo_mapa(mapa_original_norm,imagem_binaria_sem_artefatos)    
   
  maior_valor = int(distance)
  print(maior_valor)
       
  mapa_expandido_original = expandir_mapas_do_tamanho_do_tracado(mapa_original_norm,maior_valor) 
    
  vaso_expandido_bin = expandir_mapas_do_tamanho_do_tracado(imagem_binaria_sem_artefatos,maior_valor)       
      
  linha_offset_esquerda, linha_central,linha_offset_direita, maior_tamanho = retorna_linhas_offset_posicao_tamanho(array_medial_path,distancia)  
  
  #Criação das linhas à direita, centro e à esquerda a partir do traçado originado pelas curvas de beizier
  dst_array_np = retorna_dst_array_np(linha_offset_esquerda, linha_central,linha_offset_direita, maior_tamanho)

  
  #Execução do algoritmo que faz a transformação do mapa expandido
  img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = rotacionando_mapa_expandido(mapa_expandido_original,dst_array_np,maior_tamanho)     
  #Máscara do mapa
  mask_map = criar_mascara_binaria_mapa(new_dst,img_out)
  
  #Máscara do vaso
  mask_vaso = criar_mascara_binaria_vaso(vessel_map,new_origin,array_medial_path,img_out)   
  
  #Vaso binário rotacionado
  vaso_binario_rotacionado = criar_vaso_binario_expandido(vaso_expandido_bin,dst_array_np,maior_tamanho)   
  
  #Mapa sem artefatos
  mapa_sem_artefatos = retirar_artefatos(img_out,mask_map)  
  
  #Vaso binário expandido e rotacionado VER AQUI
  img_out_bin = criar_vaso_binario_expandido(vaso_expandido_bin,dst_array_np,maior_tamanho)
  
  #Vaso binário rotacionado sem artefatos
  vaso_sem_artefatos = retirar_artefatos(img_out_bin,mask_vaso)    
  
  try:
    mapa_sem_artefatos = transf_map_dist(mapa_sem_artefatos,mask_map,vaso_sem_artefatos,back_artif)
  except:
    pass  

  return  vaso_sem_artefatos,mapa_sem_artefatos,mask_map, limiar1


   