import tracemalloc, time, os
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import functions


# Windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Linux
# root_dir = f"/home/adriano/projeto_mestrado/modules"
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

from Background import background_generation as backgen

img_dir = f'{root_dir}/Images/retina/cutouts/tests'
img_label = f'{root_dir}/Images/retina/cutouts/labels'

images_vector = functions.read_directories(img_dir)
label_vector = functions.read_directories(img_label)


# Number of desired backgrounds
amounts = 1

#for i in range(len(images_vector)):
for i in range(20):

    img = np.array(Image.open(f'{img_dir}/{images_vector[i]}'))
    label = np.array(Image.open(f'{img_label}/{label_vector[i]}'))

    # Measuring the amount of memory used in the creation of the artificial background
    tracemalloc.start()

    # Measuring the time spent
    start_time = time.time()

    # Calling the function that creates the backgrounds
    generated_background = backgen.estimate_background(img, label)
    background_name = f'{root_dir}/Images/retina/backgrounds/teste_8/{i+1}.tiff'
    
    # Converting to Pillow Image
    background_image = Image.fromarray(generated_background)

    # Saving with Pillow
    background_image.save(background_name)
    
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    execution_time = end_time - start_time
