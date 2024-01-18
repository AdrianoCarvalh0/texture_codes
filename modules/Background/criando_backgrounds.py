import tracemalloc, time, os
import numpy as np
from PIL import Image
from pathlib import Path

import background_generation as backgen

# windows
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# linux
#root_dir = f"/home/adriano/projeto_mestrado/modules"

img_dir = root_dir + '/Imagens/vessel_data/images'
lab_dir = root_dir + '/Imagens/vessel_data/labels_20x'

#Varrendo o diretório
filenames = []
for filename in os.listdir(img_dir):     
    filenames.append(filename.split('.')[0])

# ordenação do array
filenames.sort()

# quantidade de backgrounds desejados
qtde = 100

for i in range(qtde):
    
    img = np.array(Image.open(img_dir + f'/{filenames[i]}.tiff'))
    label = np.array(Image.open(lab_dir + f'/{filenames[i]}.png'))

    # Mensurando a quantidade de memória utilizada na criação do background artifcial
    tracemalloc.start()

    # Mensuração do tempo gasto
    start_time = time.time()

    # chamando a função que cria os backgrounds
    generated_background = backgen.estimate_background(img, label)
    nome_background = f'{root_dir}/Background/Mapas_gerados_artificialmente/{filenames[i]}.tiff'
    
    # transformando em Image pillow
    back = Image.fromarray(generated_background)

    # salvando com o pillow
    back.save(nome_background)
    
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    execution_time = end_time - start_time