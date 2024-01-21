import numpy as np
import slice_mapper_util as smutil

def encontrar_mediana_fundo_mapa(img,img_label):    
    ints_fundo_mapa = img[img_label==0]
    threshold = np.median(ints_fundo_mapa)
    return threshold

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
  
def inserindo_vaso_no_fundo(img,img_label,backg,point):
  
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

def merge(img_fundo,img_mapa,mask_vaso,p):
   img_map_copy = img_mapa.copy()
   pixeis =  np.nonzero(mask_vaso==0)
   num_pix = int(len(pixeis)*p)
   inds = np.random.choice(range(len(pixeis)), size=num_pix, replace=False)
   pixeis_replace_x = pixeis[0][inds]
   pixeis_replace_y = pixeis[1][inds]

   img_map_copy[pixeis_replace_x,pixeis_replace_y] = img_fundo[pixeis_replace_x,pixeis_replace_y]

   return img_map_copy

def plot_fill_means_std_dev_normal_all(intensities_common_axis):
    """Function that plots all normalized intensities, displaying the difference between the mean and standard deviation
    across intensities.

    Parameters:
    -----------
    intensities_common_axis: ndarray, float
        Vector containing normalized intensities.
    Returns:
    -----------
        Plots all normalized intensities, displaying the difference between the mean and standard deviation
        across intensities.
    """
    # Returns the mean of all mapped values along the columns
    means = np.mean(intensities_common_axis, axis=0)

    # Returns the standard deviation of all mapped values along the columns
    std_dev = np.std(intensities_common_axis, axis=0)

    plt.figure(figsize=[12, 10])
    plt.title("Filling between the mean intensity and standard deviation along the columns, with the axis normalized")

    # Shows the shading
    plt.fill_between(range(len(means)), means - std_dev, means + std_dev, alpha=0.3)

    # Shows the mean
    plt.plot(range(len(means)), means)
    #plt.savefig('plot_fill_means_std_dev_normal_all.pdf')
    plt.show()