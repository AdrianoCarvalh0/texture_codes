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

# Image file names for different artificially generated backgrounds
background1 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 1-20X.tiff'
background2 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@41-Image 4-20X.tiff'
background3 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@46-Image 2-20X.tiff'
background4 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@59-Image 4-20X.tiff'


# Load artificially generated background images
background_artif1 = np.array(Image.open(f'{root_dir}/Background/Artificially_generated_maps/{background1}'))
background_artif2 = np.array(Image.open(f'{root_dir}/Background/Artificially_generated_maps/{background2}'))
background_artif3 = np.array(Image.open(f'{root_dir}/Background/Artificially_generated_maps/{background3}'))
background_artif4 = np.array(Image.open(f'{root_dir}/Background/Artificially_generated_maps/{background4}'))

# Plot and save the images
plt.figure(figsize=[10, 10])
plt.imshow(background_artif1, 'gray', vmin=0, vmax=60)
#plt.savefig('back_artif1.svg', format='svg')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_artif2, 'gray', vmin=0, vmax=60)
#plt.savefig('back_artif2.svg', format='svg')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_artif3, 'gray', vmin=0, vmax=60)
#plt.savefig('back_artif3.svg', format='svg')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_artif4, 'gray', vmin=0, vmax=60)
#plt.savefig('back_artif4.svg', format='svg')
plt.show()
