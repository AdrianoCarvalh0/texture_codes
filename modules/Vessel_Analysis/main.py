import sys
from pathlib import Path
# linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

# path windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

import vessel_analysis as va


if __name__ == '__main__':
    '''Calls the function that reads all .json files 
       and transforms them into a vessel model by writing to a folder designated as the output directory'''
    
    parameters ={
        'dir_json': f'{root_dir}/Extracted_json_vectors/retina',   # directory of jscon vectors    
        'extension': '.tif',   # image's extension    
        'dir_images': f'{root_dir}/Images/retina/images',  # original images directory              
        'out_dir_save_data': f'{root_dir}/Vessel_models_pickle/retina',  # output directory of save data      
    }
    va.generate_vessel_models(parameters)