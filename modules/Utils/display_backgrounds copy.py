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
background2 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@75-Image 3-20X.tiff'
#background3 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@60-Image 1-20X.tiff'
#background4 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@65-Image 3-20X.tiff'
#background5 = 'T-3 Weeks@Females@397 F@397-stroke-middle-20X-03.tiff'
#background6 = 'T-3 Weeks@Females@391 F@391-CTL-Middle-20X-01.tiff'



# Load artificially generated background images
background_artif1 = np.array(Image.open(f'{root_dir}/Training_validation/Original_Images/{background1}'))
background_artif2 = np.array(Image.open(f'{root_dir}/Training_validation/Original_Images/{background2}'))
#background_artif3 = np.array(Image.open(f'{root_dir}/Training_validation/Original_Images/{background3}'))
#background_artif4 = np.array(Image.open(f'{root_dir}/Training_validation/Original_Images/{background4}'))
#background_artif5 = np.array(Image.open(f'{root_dir}/Training_validation/Original_Images/{background5}'))
#background_artif6 = np.array(Image.open(f'{root_dir}/Training_validation/Original_Images/{background6}'))

# Plot and save the images
plt.figure(figsize=[10, 10])
plt.imshow(background_artif1, 'gray', vmin=0, vmax=100)
plt.savefig('back_artif1.svg', format='svg')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_artif2, 'gray', vmin=0, vmax=100)
plt.savefig('back_artif2.svg', format='svg')
plt.show()

'''plt.figure(figsize=[10, 10])
plt.imshow(background_artif3, 'gray', vmin=0, vmax=100)
plt.savefig('back_artif3.svg', format='svg')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_artif4, 'gray', vmin=0, vmax=100)
plt.savefig('back_artif4.svg', format='svg')
plt.show()


plt.figure(figsize=[10, 10])
plt.imshow(background_artif5, 'gray', vmin=0, vmax=100)
plt.savefig('back_artif5.svg', format='svg')
plt.show()

plt.figure(figsize=[10, 10])
plt.imshow(background_artif6, 'gray', vmin=0, vmax=100)
plt.savefig('back_artif6.svg', format='svg')
plt.show()'''