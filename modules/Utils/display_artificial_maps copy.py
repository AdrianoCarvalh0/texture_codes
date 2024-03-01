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
background1 = f"{root_dir}/Training_validation/Artificial_images/Generated_from_5_maps/pack5/images/Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@60-Image 2-20X_18_with_20.tiff"
background2 = f"{root_dir}/Training_validation/Artificial_images/Generated_from_5_maps/pack9/images/Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@41-Image 2-20X_20_with_20.tiff"
background3 = f"{root_dir}/Training_validation/Artificial_images/Generated_from_5_maps/pack4/images/Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@75-Image 4-20X_16_with_49.tiff"
background4 = f"{root_dir}/Training_validation/Artificial_images/Generated_from_5_maps/pack4/images/Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@60-Image 1-20X_77_with_49.tiff"




# Load images with vessels from different backgrounds and directories
background_with_vessels1 = np.array(Image.open(f'{background1}'))
background_with_vessels2 = np.array(Image.open(f'{background2}'))
background_with_vessels3 = np.array(Image.open(f'{background3}'))
background_with_vessels4 = np.array(Image.open(f'{background4}'))

# Plot and save the images
plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels1, 'gray', vmin=0, vmax=255)
plt.axis('off')
plt.savefig('background_with_vessels1.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels2, 'gray', vmin=0, vmax=189)
plt.axis('off')
plt.savefig('background_with_vessels2.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels3, 'gray', vmin=0, vmax=189)
plt.axis('off')
plt.savefig('background_with_vessels3.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_with_vessels4, 'gray', vmin=0, vmax=255)
plt.axis('off')
plt.savefig('background_with_vessels4.svg', format='svg')
plt.show()

