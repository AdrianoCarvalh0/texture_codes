import background_generation as backgen
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import sys
from matplotlib import pyplot as plt

import background_generation as backgen
#windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

#linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

from Utils import functions


def return_background(generate,dir_images,dir_labels,directory_backs,vector_backgrounds,img_back):
    if generate:
        while not img_back in vector_backgrounds:
            if img_back in dir_images and img_back in dir_labels:
                background = backgen.estimate_background(f'{dir_images}/{img_back}', f'{dir_labels}/{img_back}')
            else:
                n_background = np.random.randint(0, len(vector_backgrounds))
                img_back = vector_backgrounds[n_background]
                background = np.array(Image.open(f'{directory_backs}/{vector_backgrounds[n_background]}')) 
    else:
        n_background = np.random.randint(0, len(vector_backgrounds))
        img_back = vector_backgrounds[n_background]
        background = np.array(Image.open(f'{directory_backs}/{img_back}'))
    return background 

def returns_array_pickle(num_maps,array_maps_pickle):
    sorted_array_pickels = []
    for i in range(num_maps):
        n_pickle = np.random.randint(1, len(array_maps_pickle))    
        sorted_array_pickels.append(array_maps_pickle[n_pickle])
    return sorted_array_pickels

def compatible_map_with_backg(sorted_array_pickels, array_backrounds, directory_backs,dir_maps_pickle):
    n_background = np.random.randint(0, len(array_backrounds))
    img_back = array_backrounds[n_background]
    background = np.array(Image.open(f'{directory_backs}/{img_back}'))
    cont = 0
    for i in range(len(sorted_array_pickels)):
        path_map = (f"{dir_maps_pickle}/{sorted_array_pickels[i]}")
        map_pickle = pickle.load(open(path_map, 'rb'))           
        vessel_map = map_pickle['vessel_model'].vessel_map 
        original_map = vessel_map.mapped_values
        vessel_mask = vessel_map.mapped_mask_values
        normalized_original_map = backgen.normalize(background,original_map,vessel_mask,30)            
        if normalized_original_map is not None:
            cont += 1                  
    if cont == len(sorted_array_pickels):
        return img_back
    else:
        return None

def check_compatible(array_pickles,number_images,array_backrounds, directory_backs,dir_maps_pickle):
    count_errors = 0
    vector_names = []
        
    for i in range(number_images):
        name =  compatible_map_with_backg(array_pickles, array_backrounds, directory_backs, dir_maps_pickle)
        if name is not None:        
            vector_names.append(name)
        else:
            count_errors += 1       
    print(f"incompatible: {count_errors}")
    return vector_names


def generate_maps(params):
    array_maps_pickle = functions.read_directories(params['dir_maps_pickle'])
    array_images = functions.read_directories(params['dir_images'])
    array_labels = functions.read_directories(params['dir_labels'])
    array_backrounds = functions.read_directories(params['dir_backs'])
    array_traces = functions.read_directories(params['dir_traces'])

    number_maps = params['num_maps']
    dir_maps_pickle = params['dir_maps_pickle']
    directory_backs = params['dir_backs']
    directory_traces =  params['dir_traces']
    directory_out = params['out_dir']
    num_images = params['num_images']
    min_number_vessels = params['min_number_vessels']
    max_number_vessels = params['max_number_vessels']    

    array_maps_pickle_sorted = returns_array_pickle(number_maps,array_maps_pickle)    

    #compatible_map_with_backg(array_maps_pickle_sorted, array_backrounds, directory_backs,dir_maps_pickle)

    vector_names_background = check_compatible(array_maps_pickle, num_images,array_backrounds,directory_backs,dir_maps_pickle)   

    none_results = 0

    for j in range(num_images):
        number_of_vessels = np.random.randint(min_number_vessels, max_number_vessels)        
        
        n_background = np.random.randint(0, len(vector_names_background))
        name_background = vector_names_background[n_background]
        background = np.array(Image.open(f'{directory_backs}/{name_background}'))          
    
        background_name = name_background.replace("'","").replace(".tiff","")

        clipping_background = background[0:1100,0:1370]
        background_with_pad = np.pad(clipping_background, ((200,200),(200,200)), mode="symmetric", reflect_type="even")
        background_bin = np.zeros(background_with_pad.shape)

        background_with_vessels_bin = background_bin.copy()
        background_with_vessels = background_with_pad.copy()

        has_maps =  np.full(shape = background_with_pad.shape, fill_value=0)
        has_maps_bin =  np.full(shape = background_bin.shape, fill_value=0)

        counter = 0
        while counter < number_of_vessels:
            n_traces = np.random.randint(0, len(array_traces))
            trace = array_traces[n_traces]    
            vector_medial_path = backgen.return_paths(f"{directory_traces}/{trace}")           
            results = backgen.insert_vessels(vector_medial_path[0], vector_medial_path[1], array_maps_pickle_sorted,dir_maps_pickle,background,30)        
            if results is not None:
                vessel_without_artifacts, map_without_artifacts, mask_map, treshold = results  
                background_with_vessels = backgen.insert_map(background_with_vessels,vessel_without_artifacts,map_without_artifacts,mask_map, treshold, has_maps)
                background_with_vessels_bin = backgen.insert_binary_map(background_with_vessels_bin,vessel_without_artifacts,has_maps_bin)
                counter +=1
            else:
                none_results += 1  

        background_clipped = background_with_vessels[200:1304,200:1576]
        background_clipped_bin = background_with_vessels_bin[200:1304,200:1576]    

        img1 = Image.fromarray(background_clipped.astype(np.uint8))
        path = f"{directory_out}/images/{background_name}_{j}_with_{number_of_vessels}.tiff"
        img = img1.save(path)

        img2 = Image.fromarray(background_clipped_bin.astype(np.bool_))
        path = f"{directory_out}/labels/{background_name}_{j}_with_{number_of_vessels}.tiff"
        img = img2.save(path)

if __name__ == '__main__':

    #Aumentar o limiar para testar se fica compatível no máximo valor
    #Gerar as curvas de bezier dentro do algoritmo - tamanho setado pelo cliente - distancia entre o ponto 1 e ponto 2, numero de pontos, max_vd, 
    parameters = { 'dir_maps_pickle': f'{root_dir}/Vessel_models_pickle',
                'num_maps': 5, # number of maps to be inserted
                'num_images': 100, #number of images desired
                'dir_backs': f'{root_dir}/Background/Artificially_generated_maps', #background's directory
                'dir_images': f'{root_dir}/Images/vessel_data/images', # original images directory
                'dir_labels': f'{root_dir}/Images/vessel_data/labels_20x', # label directory
                'dir_traces': f'{root_dir}/Artificial_lines/bezier_traces', # directory of traces - Bezier curves
                'generate_back':False, #
                'out_dir': f'{root_dir}/Images/Background_with_vessels_tests', #output directory
                'min_number_vessels': 2, # minimum number of vessels
                'max_number_vessels': 5, # maximum number of vessels
                # Bezier Curves parameters
                'max_vd' : 500,
                'precision': 100,
                'number_cols': 1776,
                'number_rows': 1504,
                'number_points': 25,
            }    
    generate_maps(parameters)