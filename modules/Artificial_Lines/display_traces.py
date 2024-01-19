from matplotlib import pyplot as plt
import sys, os

#linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
root_dir = f"/home/adriano/projeto_mestrado/modules"

#windows
#sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
#root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

bezier_traces_tests = f'{root_dir}/Artificial_Lines/bezier_traces_tests/'

filenames = []
for filename in os.listdir(bezier_traces_tests):
     filenames.append(filename) 

traces_array = filenames

from Background import background_generation as backgen
from geopandas import gpd

for i in range(len(traces_array)):
    array_medial_path = backgen.retorna_paths(f'{bezier_traces_tests}/{traces_array[i]}')   
    line_offset_left, line_central,line_offset_right, max_valor = backgen.returns_lines_offset_position_size(array_medial_path[0],30)   
    fig, ax2 = plt.subplots(figsize=(10,5))
    gp4 = gpd.GeoSeries([line_offset_left, line_central, line_offset_right])   
    gp4.plot(ax=ax2, cmap="tab10")   
    #fig.savefig(f'{root_dir}/Artificial_Lines/LineStrings/varios_pontos_controle/tracado_{i}.svg', format='svg')