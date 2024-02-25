from matplotlib import pyplot as plt
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Linux
# sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
# root_dir = f"/home/adriano/projeto_mestrado/modules"

# Windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Image file names for different backgrounds with vessels
background1 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 1-20X_77_com_29.tiff'
background2 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 2-20X_69_com_36.tiff'
background3 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 2-20X_92_com_41.tiff'
background4 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 3-20X_90_com_28.tiff'

# Directories for artificial images generated from different numbers of maps
train_dir = 'Training_validation/Artificial_images/Generated_from_1_map/pack5/images'


# Load images with vessels from different backgrounds and directories
background_with_vessels1 = np.array(Image.open(f'{root_dir}/{train_dir}/{background1}'))
background_with_vessels2 = np.array(Image.open(f'{root_dir}/{train_dir}/{background2}'))
background_with_vessels3 = np.array(Image.open(f'{root_dir}/{train_dir}/{background3}'))
background_with_vessels4 = np.array(Image.open(f'{root_dir}/{train_dir}/{background4}'))

# Plot and save the images
plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels1, 'gray', vmin=0, vmax=127)
plt.savefig('background_with_vessels1.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels2, 'gray', vmin=0, vmax=127)
plt.savefig('background_with_vessels2.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels3, 'gray', vmin=0, vmax=127)
plt.savefig('background_with_vessels3.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels4, 'gray', vmin=0, vmax=127)
plt.savefig('background_with_vessels4.svg', format='svg')
plt.show()

