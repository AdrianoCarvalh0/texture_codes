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

def compatible_map_with_backg(sorted_array_pickels, array_backrounds, directory_backs,dir_maps_pickle,treshold):
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
        normalized_original_map = backgen.normalize(background,original_map,vessel_mask,treshold)            
        if normalized_original_map is not None:
            cont += 1                  
    if cont == len(sorted_array_pickels):
        return img_back
    else:
        return None

def check_compatible(array_pickles,number_images,array_backrounds, directory_backs,dir_maps_pickle,treshold,dir_images,dir_labels):
    count_errors = 0
    vector_names = []
        
    for i in range(number_images):
        n_random = np.random.randint(0, len(array_backrounds))
        name =  compatible_map_with_backg(array_pickles, array_backrounds, directory_backs, dir_maps_pickle,treshold)
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

    number_maps = params['num_maps']
    dir_maps_pickle = params['dir_maps_pickle']
    directory_backs = params['dir_backs']
    
    directory_out = params['out_dir']
    num_images = params['num_images']
    min_number_vessels = params['min_number_vessels']
    max_number_vessels = params['max_number_vessels']
    treshold = params['treshold']

    #parameters of curve Bezier
    max_distance = params['max_distance']
    control_points = params['control_points']
    precision = params['precision']
    min_len_trace = params['min_len_trace']
    max_len_trace = params['max_len_trace']
    number_points = params['number_points']
    padding = params['padding']
    number_cols = params['number_cols']
    number_rows = params['number_rows']

    array_maps_pickle_sorted = returns_array_pickle(number_maps,array_maps_pickle)        

    vector_names_background = check_compatible(array_maps_pickle, num_images,array_backrounds,directory_backs,dir_maps_pickle,treshold)   

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
            points, distance = backgen.create_points(number_points, padding, number_rows, number_cols, min_len_trace, max_len_trace)
            curve = backgen.create_curve(points, max_distance, control_points, precision)         
            results = backgen.insert_vessels(curve, distance, array_maps_pickle_sorted,dir_maps_pickle,background,treshold)        
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
    
    parameters ={
        'dir_maps_pickle': f'{root_dir}/Vessel_models_pickle',
        'num_maps': 1,  # number of maps to be inserted
        'num_images': 5,  # number of images desired
        'dir_backs': f'{root_dir}/Background/Artificially_generated_maps',  # background's directory
        'dir_images': f'{root_dir}/Images/vessel_data/images',  # original images directory
        'dir_labels': f'{root_dir}/Images/vessel_data/labels_20x',  # label directory
        'dir_traces': f'{root_dir}/Artificial_lines/bezier_traces',  # directory of traces - Bezier curves
        'generate_back': False,  # whether to generate background images
        'out_dir': f'{root_dir}/Images/Background_with_vessels_tests',  # output directory
        'min_number_vessels': 1,  # minimum number of vessels
        'max_number_vessels': 3,  # maximum number of vessels
        'treshold': 33,  # parameter that defines the threshold between the differences of the map background and the overall background
        
        # Bezier Curves parameters
        'max_distance': 500,  # maximum distance where control points will be randomly drawn. Example: 1 generates straight lines.
        'control_points': 6,  # number of control points between pe and ps. The higher this number, the more curves are generated.
        'precision': 100,  # number of points generated
        'number_cols': 1776,  # maximum number of columns for the Bezier curve
        'number_rows': 1504,  # maximum number of rows for the Bezier curve
        'number_points': 25,  # determine the number of random points to be generated
        'min_len_trace': 500,  # minimum distance between the initial and final points
        'max_len_trace': 1300,  # maximum distance between the initial and final points
        'padding': 0,  # padding used to ensure that the trace does not exceed the size of the background
    }
    
    generate_maps(parameters)