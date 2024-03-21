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
        'dir_maps_pickle': f'{root_dir}/Training_validation/Maps/5_maps_5_images/pack10',  # directory of pickles
        'num_maps': 5,  # number of maps to be inserted
        'num_images': 50, # number of images desired
        'dir_backs': f'{root_dir}/Background/Artificially_generated_maps',  # background's directory
        'dir_images': f'{root_dir}/Images/vessel_data/images',  # original images directory
        'dir_labels': f'{root_dir}/Images/vessel_data/labels_20x',  # label directory       
        'generate_back': True,  # whether to generate background images       
        'out_dir_images': f'{root_dir}/Training_validation/Artificial_images/Generated_from_5_maps/pack10/images',  # output directory of images
        'out_dir_labels': f'{root_dir}/Training_validation/Artificial_images/Generated_from_5_maps/pack10/labels',  # output directory of labels
        'min_number_vessels': 20,  # minimum number of vessels
        'max_number_vessels': 50,  # maximum number of vessels
        'threshold': 30,  # parameter that defines the threshold between the differences of the map background and the overall background
        
        # Bezier Curves parameters
        'max_distance': 500,  # maximum distance where control points will be randomly drawn. Example: 1 generates straight lines.
        'control_points': 6,  # number of control points between pe and ps. The higher this number, the more curves are generated.
        'precision': 100,  # number of points generated
        'number_cols': 1776,  # maximum number of columns for the Bezier curve
        'number_rows': 1504,  # maximum number of rows for the Bezier curve
        'number_points': 25,  # determine the number of random points to be generated
        'min_len_trace': 500,  # minimum distance between the initial and final points
        'max_len_trace': 1300,  # maximum distance between the initial and final points
        'padding': 60,  # padding used to ensure that the trace does not exceed the size of the background
    }
    backgen.generate_backgrounds_with_vessels(parameters)