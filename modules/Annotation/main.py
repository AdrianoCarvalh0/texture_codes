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
        'image': 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 1-20X',
        'root_img': f'{root_dir}/Images/vessel_data/images',   # original images directory
        'root_out':  f'{root_dir}/Extracted_json_vectors/teste',  # Path where the JSON file will be saved
        'list_array': list_array, # array that will store the coordinates resulting from manual marking  
    }
    ann.generate_dash(parameters)