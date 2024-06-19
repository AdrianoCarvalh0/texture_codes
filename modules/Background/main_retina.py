import tracemalloc
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

if __name__ == '__main__':

    #Aumentar o limiar para testar se fica compatível no máximo valor
    #Gerar as curvas de bezier dentro do algoritmo - tamanho setado pelo cliente - distancia entre o ponto 1 e ponto 2, numero de pontos, max_vd, 
    parameters ={
        'dir_maps_pickle': f'{root_dir}/Vessel_models_pickle/retina/training',  # directory of pickles
        'dir_mask': f'{root_dir}/Images/retina/mask',  # directory of pickles
        'num_maps': 10,  # number of maps to be inserted
        'num_images': 100, # number of images desired
        'dir_backs': f'{root_dir}/Images/retina/backgrounds',  # background's directory
        #'dir_images': f'{root_dir}/Images/retina/images_training',  # original images directory
        #'dir_labels': f'{root_dir}/Images/retina/labels_training',  # label directory       
        'generate_back': False,  # whether to generate background images       
        'out_dir_images': f'{root_dir}/Training_validation_retina/Artificial_images/Generated_from_80_maps/images',  # output directory of images
        'out_dir_labels': f'{root_dir}/Training_validation_retina/Artificial_images/Generated_from_80_maps/labels',  # output directory of labels
        'min_number_vessels': 15,  # minimum number of vessels
        'max_number_vessels': 30,  # maximum number of vessels
        'threshold': 30,  # parameter that defines the threshold between the differences of the map background and the overall background
        
        # Bezier Curves parameters
        'max_distance': 500,  # maximum distance where control points will be randomly drawn. Example: 1 generates straight lines.
        'control_points': 6,  # number of control points between pe and ps. The higher this number, the more curves are generated.
        'precision': 100,  # number of points generated
        'number_cols': 900,  # maximum number of columns for the Bezier curve
        'number_rows': 900,  # maximum number of rows for the Bezier curve
        'number_points': 25,  # determine the number of random points to be generated
        'min_len_trace': 400,  # minimum distance between the initial and final points
        'max_len_trace': 900,  # maximum distance between the initial and final points
        'padding': 50,  # padding used to ensure that the trace does not exceed the size of the background
    }
    backgen.generate_backgrounds_with_vessels_retina(parameters)