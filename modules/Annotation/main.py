from pathlib import Path


#root_dir linux
#root_dir ="/home/adriano/projeto_mestrado/modules"

#root_dir windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

import annotation as ann

# Dash app main method. Creates a browser-accessible address for demarcating the vessel(s).
if __name__ == "__main__":
    list_array = []
    parameters ={
        'image': '34_training',
        'extension': '.tif',
        'root_img': f'{root_dir}/Images/retina/images',   # original images directory
        'root_out':  f'{root_dir}/Extracted_json_vectors/retina/',  # Path where the JSON file will be saved
        'list_array': list_array, # array that will store the coordinates resulting from manual marking  
    }
    ann.generate_dash(parameters)