from matplotlib import pyplot as plt
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# Linux
# sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
# root_dir = f"/home/adriano/projeto_mestrado/modules"

# Windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")


from Background import background_generation as backgen

from Utils import functions

dir_backs = f'{root_dir}/Images/retina/backgrounds'

back_vector = functions.read_directories(dir_backs)


for i in range(len(back_vector)):

    # Plot and save the images

    path_img = back_vector[i]
    background = np.array(Image.open(f'{dir_backs}/{path_img}'))   
    print(f'Background: "{path_img}')
    print(f'Shape: {background.shape}')
    #plt.figure(figsize=[10, 10])
    #plt.title({path_img})
    #plt.imshow(background, 'gray')
    #plt.savefig('back_artif1.svg', format='svg')
    #plt.show()


