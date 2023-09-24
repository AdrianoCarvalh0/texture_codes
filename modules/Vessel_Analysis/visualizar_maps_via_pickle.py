import sys, pickle
from matplotlib import pyplot as plt

sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

from Funcoes_gerais import funcoes


root_dir = f"/home/adriano/projeto_mestrado/modules"
img_dir = f'{root_dir}/Imagens/vessel_data/images/'
lab_dir = f'{root_dir}/Imagens/vessel_data/labels_20x/'
pickle_dir = f'{root_dir}/Vessel_Models_pickle/novos'

vetor_pickles = funcoes.ler_diretorios(pickle_dir)


for i in range(len(vetor_pickles)):
    path = (pickle_dir + f'/{vetor_pickles[i]}')

    arquivo_pickle = pickle.load(open(path, 'rb')) 
    vessel_map = arquivo_pickle['vessel_model'].vessel_map 
    mapa_original = vessel_map.mapped_values
    print(f'/{vetor_pickles[i]}')
    plt.figure(figsize=[10, 8])
    plt.title("mapa_original")
    plt.imshow(mapa_original, 'gray', vmin=0, vmax=60)
    plt.plot()