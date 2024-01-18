from matplotlib import pyplot as plt
import sys, os, json
from pathlib import Path
#linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

#windows
#sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Funcoes_gerais import funcoes

# linux
root_dir = f"/home/adriano/projeto_mestrado/modules"

# windows
#root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

tracados_bezier = f'{root_dir}/Artificial_Lines/tracados_bezier_testes/'
teste = '/home/adriano/projeto_mestrado/modules/Artificial_Lines/tracados_bezier_testes'

filenames = []
for filename in os.listdir(teste):
     filenames.append(filename) 

array_tracados = filenames

from Background import background_generation as backgen
from geopandas import gpd

for i in range(len(array_tracados)):
    vetor_medial_path = backgen.retorna_paths(f'{tracados_bezier}/{array_tracados[i]}')   
    linha_offset_esquerda, linha_central,linha_offset_direita, maior_tamanho = backgen.retorna_linhas_offset_posicao_tamanho(vetor_medial_path[0],30)   
    fig, ax2 = plt.subplots(figsize=(10,5))
    gp4 = gpd.GeoSeries([linha_offset_esquerda, linha_central, linha_offset_direita])   
    gp4.plot(ax=ax2, cmap="tab10")   
    #fig.savefig(f'{root_dir}/Artificial_Lines/LineStrings/varios_pontos_controle/tracado_{i}.svg', format='svg')