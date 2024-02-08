from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# linux
#ys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

# windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")


from Slice_mapper import slice_mapper_util as smutil


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

    def plot_figure2(img, vessel_model, cross_paths):       
      """Function that creates the vessel model and transversal paths.

      Parameters:
      -----------
      img: ndarray, float
          Resized image containing the area of the extracted vessel.
      vessel_model: object VesselModel
          Returns the vessel model with an instantiated object of the VesselModel class.
      cross_paths: ndarray, float
          Transversal paths.
      Returns:
      -----------
          Plots the resized image, along with the vessel model, transversal paths, and translated paths 1 and 2,
          in three different ways:
          1 - with the mapped values having the minimum at 0 and maximum at 60
          2 - values mapped in the standard range, from 0 to 255
          3 - values mapped between the minimum 0 and maximum in the values found in the mapping
      """

      vessel_map = vessel_model.vessel_map
      fig = plt.figure(figsize=[12, 10])
      ax = fig.add_subplot()
      slice_mapper.plot_model(img, vessel_model, cross_paths, ax)        
      norm = ax.images[0].norm
      norm.vmin, norm.vmax = 0, 60
      
      
      plt.figure(figsize=[12, 10])    
      plt.plot()
      plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)
      plt.plot(vessel_map.path1_mapped, c='green')
      plt.plot(vessel_map.path2_mapped, c='green')
      #file_min0max60 = f'{master_folder}/Images/plots/min0max60/{imag}_{x}.png'
      #plt.savefig(file_min0max60)

      plt.figure(figsize=[12, 10])    
      plt.plot()
      plt.imshow(vessel_map.mapped_values[::-1], 'gray', vmin=0, vmax=255)
      plt.plot(vessel_map.path1_mapped, c='green')
      plt.plot(vessel_map.path2_mapped, c='green')    
      #file_min0max255 = f'{master_folder}/Images/plots/min0max255/{imag}_{x}.png'
      #plt.savefig(file_min0max255)

      plt.figure(figsize=[12, 10])   
      plt.plot()
      plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=vessel_map.mapped_values.max())     
      plt.plot(vessel_map.path1_mapped, c='green')
      plt.plot(vessel_map.path2_mapped, c='green')    
      #file_min0max = f'{master_folder}/Images/plots/min0maxmapped/{imag}_{x}.png'
      #plt.savefig(file_min0max)


def insert_vessels2(medial_path_array, distance, vessel_map, normalized_original_map, binary_map_without_artifacts, back_artif):   
    
    # Get dimensions of the original map
    rows, cols = normalized_original_map.shape

    # Calculate the distance to be used for the map expansion
    distance = (rows / 2)

    # Find the threshold based on the most frequent pixel value in the normalized map
    threshold1 = find_most_frequent_pixel(normalized_original_map)

    # Set the maximum value for expansion
    max_value = int(distance)

    # Expand the original map to the specified size
    expanded_original_map = expand_maps_to_trace_size(normalized_original_map, max_value)

    # Expand the binary vessel map to the specified size
    expanded_vessel_bin = expand_maps_to_trace_size(binary_map_without_artifacts, max_value)

    # Get left, central, and right lines for the expansion from the medial path
    left_offset_line, central_line, right_offset_line, max_size = returns_lines_offset_position_size(
        medial_path_array, distance)

    # Create destination array for transformation from left, central, and right lines
    dst_array_np = returns_dst_array_np(left_offset_line, central_line, right_offset_line, max_size)

    # Execute the algorithm that transforms the expanded map
    img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = rotate_expanded_map(
        expanded_original_map, dst_array_np, max_size)

    # Create a binary mask for the map
    mask_map = create_binary_mask_map(new_dst, img_out)

    # Create a binary mask for the vessel
    mask_vessel = create_binary_mask_vessel(vessel_map, new_origin, medial_path_array, img_out)

    # Remove artifacts from the map
    map_without_artifacts = remove_artifacts(img_out, mask_map)

    # Get dimensions of the map without artifacts
    rows_art, cols_art = map_without_artifacts.shape

    # Get dimensions of the artificial background
    rows_back, cols_back = back_artif.shape

    #import pdb; pdb.set_trace()
    # Return None if the dimensions of the map without artifacts are greater than or equal to the background artifact
    #if rows_art >= rows_back or cols_art >= cols_back:
        #return None

    # Create an expanded and rotated binary vessel map
    img_out_bin = create_binary_expanded_vessel(expanded_vessel_bin, dst_array_np, max_size)

    # Remove artifacts from the binary vessel map
    vessel_without_artifacts = remove_artifacts(img_out_bin, mask_vessel)

    # Transform the map without artifacts
    map_without_artifacts_transf = transform_map_dist2(map_without_artifacts, mask_map, vessel_without_artifacts,
                                                        back_artif)

    # Return the result if the transformed map without artifacts is not None
    if map_without_artifacts_transf is not None:
        return vessel_without_artifacts, map_without_artifacts_transf, mask_map, threshold1
    else:
        return vessel_without_artifacts, map_without_artifacts, mask_map, threshold1
    