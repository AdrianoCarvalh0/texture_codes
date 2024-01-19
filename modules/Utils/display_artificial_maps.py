from matplotlib import pyplot as plt
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Image file names for different backgrounds with vessels
background1 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@75-Image 3-20X_21_com_49.tiff'
background2 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@74-Image 1-20X_32_com_20.tiff'
background3 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@64-Image 1-20X_39_com_49.tiff'
background4 = 'T-3 Weeks@Females@397 F@397-stroke-middle-20X-03_22_com_20.tiff'

# Windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Linux
# root_dir = f"/home/adriano/projeto_mestrado/modules"

# Directories for artificial images generated from different numbers of maps
train_dir1 = 'Training_validation/Artificial_Images\Generated_from_1_map\pack5\artificial_images'
train_dir2 = 'Training_validation/Artificial_Images\Generated_from_5_maps\artificial_images'
train_dir3 = 'Training_validation/Artificial_Images\Generated_from_10_maps\artificial_images'
train_dir4 = 'Training_validation/Artificial_Images\Generated_from_160_maps\artificial_images'

# Load images with vessels from different backgrounds and directories
background_with_vessels1 = np.array(Image.open(f'{root_dir}/{train_dir1}/{background1}'))
background_with_vessels2 = np.array(Image.open(f'{root_dir}/{train_dir2}/{background2}'))
background_with_vessels3 = np.array(Image.open(f'{root_dir}/{train_dir3}/{background3}'))
background_with_vessels4 = np.array(Image.open(f'{root_dir}/{train_dir4}/{background4}'))

# Plot and save the images
plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels1, 'gray', vmin=0, vmax=60)
plt.savefig('background_with_vessels1.tiff')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels2, 'gray', vmin=0, vmax=60)
plt.savefig('background_with_vessels2.tiff')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels3, 'gray', vmin=0, vmax=60)
plt.savefig('background_with_vessels3.tiff')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels4, 'gray', vmin=0, vmax=60)
plt.savefig('background_with_vessels4.tiff')
plt.show()
