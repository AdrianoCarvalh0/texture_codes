from matplotlib import pyplot as plt
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from pathlib import Path

back1 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 1-20X.tiff'
back2 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@41-Image 4-20X.tiff'
back3 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@46-Image 2-20X.tiff'
back4 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@59-Image 4-20X.tiff'

#windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

#linux
#root_dir = f"/home/adriano/projeto_mestrado/modules"

back_artif1 = np.array(Image.open(f'{root_dir}/Background/Mapas_gerados_artificialmente/{back1}'))
back_artif2 = np.array(Image.open(f'{root_dir}/Background/Mapas_gerados_artificialmente/{back2}'))
back_artif3 = np.array(Image.open(f'{root_dir}/Background/Mapas_gerados_artificialmente/{back3}'))
back_artif4 = np.array(Image.open(f'{root_dir}/Background/Mapas_gerados_artificialmente/{back4}'))


plt.figure(figsize=[10, 10])
plt.imshow(back_artif1, 'gray', vmin=0, vmax=60)
plt.savefig('back_artif1.svg', format='svg')
plt.plot()

plt.figure(figsize=[10, 10])
plt.imshow(back_artif2, 'gray', vmin=0, vmax=60)
plt.savefig('back_artif2.svg', format='svg')
plt.plot()

plt.figure(figsize=[10, 10])
plt.imshow(back_artif3, 'gray', vmin=0, vmax=60)
plt.savefig('back_artif3.svg', format='svg')
plt.plot()

plt.figure(figsize=[10, 10])
plt.imshow(back_artif4, 'gray', vmin=0, vmax=60)
plt.savefig('back_artif4.svg', format='svg')
plt.plot()


