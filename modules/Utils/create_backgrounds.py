import tracemalloc, time, os
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Linux
# root_dir = f"/home/adriano/projeto_mestrado/modules"
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

from Background import background_generation as backgen

img_dir = root_dir / 'Images/vessel_data/images'
lab_dir = root_dir / 'Images/vessel_data/labels_20x'

# Traversing the directory
filenames = []
for filename in os.listdir(img_dir):
    filenames.append(filename.split('.')[0])

# Sorting the array
filenames.sort()

# Number of desired backgrounds
amounts = 100

for i in range(amounts):
    '''Creates and saves as many backgrounds as set'''
    
    img = np.array(Image.open(img_dir / f'{filenames[i]}.tiff'))
    label = np.array(Image.open(lab_dir / f'{filenames[i]}.png'))

    # Measuring the amount of memory used in the creation of the artificial background
    tracemalloc.start()

    # Measuring the time spent
    start_time = time.time()

    # Calling the function that creates the backgrounds
    generated_background = backgen.estimate_background(img, label)
    background_name = f'{root_dir}/Background/Artificially_generated_maps/{filenames[i]}.tiff'
    
    # Converting to Pillow Image
    background_image = Image.fromarray(generated_background)

    # Saving with Pillow
    background_image.save(background_name)
    
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    execution_time = end_time - start_time
