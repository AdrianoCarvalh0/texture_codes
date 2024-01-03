from matplotlib import pyplot as plt
from pathlib import Path
import sys
import numpy as np
from PIL import Image

fundo1 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@75-Image 3-20X_21_com_49.tiff'
fundo2 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@74-Image 1-20X_32_com_20.tiff'
fundo3 = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@64-Image 1-20X_39_com_49.tiff'
fundo4 = 'T-3 Weeks@Females@397 F@397-stroke-middle-20X-03_22_com_20.tiff'


root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

trein_dir1 = 'Treinamento_validacao/Imagens_Artificiais\Geradas_a_partir_de_1_mapa\pack5\imagens_artificiais'
trein_dir2 = 'Treinamento_validacao/Imagens_Artificiais\Geradas_a_partir_de_5_mapas\imagens_artificiais'
trein_dir3 = 'Treinamento_validacao/Imagens_Artificiais\Geradas_a_partir_de_10_mapas\imagens_artificiais'
trein_dir4 = 'Treinamento_validacao/Imagens_Artificiais\Geradas_a_partir_de_160_mapas\imagens_artificiais'

fundo_com_vasos1 = np.array(Image.open(f'{root_dir}/{trein_dir1}/{fundo1}'))
fundo_com_vasos2 = np.array(Image.open(f'{root_dir}/{trein_dir2}/{fundo2}'))
fundo_com_vasos3 = np.array(Image.open(f'{root_dir}/{trein_dir3}/{fundo3}'))
fundo_com_vasos4 = np.array(Image.open(f'{root_dir}/{trein_dir4}/{fundo4}'))


plt.figure(figsize=[10, 10])
plt.imshow(fundo_com_vasos1, 'gray', vmin=0, vmax=60)
plt.savefig('fundo_com_vasos1.tiff')
plt.plot()

plt.figure(figsize=[10, 10])
plt.imshow(fundo_com_vasos2, 'gray', vmin=0, vmax=60)
plt.savefig('fundo_com_vasos2.tiff')
plt.plot()

plt.figure(figsize=[10, 10])
plt.imshow(fundo_com_vasos3, 'gray', vmin=0, vmax=60)
plt.savefig('fundo_com_vasos3.tiff')
plt.plot()

plt.figure(figsize=[10, 10])
plt.imshow(fundo_com_vasos4, 'gray', vmin=0, vmax=60)
plt.savefig('fundo_com_vasos4.tiff')
plt.plot()


