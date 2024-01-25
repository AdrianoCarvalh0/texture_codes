from pathlib import Path
import numpy as np
from PIL import Image
import sys
from matplotlib import pyplot as plt

import background_generation as backgen
#windows
#sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
#root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

#linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
root_dir = f"/home/adriano/projeto_mestrado/modules"

from Utils import functions


img_dir = f'{root_dir}/Images/vessel_data/images'
lab_dir = f'{root_dir}/Images/vessel_data/labels_20x'
training_dir = f'{root_dir}/Training_validation'

pickle_dir_5 = f'{training_dir}/Maps/5_maps_de_images'
pickle_dir_10 = f'{training_dir}/Maps/10_maps_10_images'
pickle_dir_40 = f'{training_dir}/Maps/160_maps_40_images'
pickle_dir_50 = f'{training_dir}/Maps/200_maps_50_images'
pickle_dir_343 = f'{root_dir}/Vessel_models_pickle'

background_dir = f'{root_dir}/Background/Artificially_generated_maps'
background_dir_5 = f'{training_dir}/Backgrounds/5_backgrounds'
background_dir_10 = f'{training_dir}/Backgrounds/10_backgrounds'
background_dir_40 = f'{training_dir}/Backgrounds/40_backgrounds'
background_dir_50 = f'{training_dir}/Backgrounds/50_backgrounds'


# 1 - Gerar fundos a partir de 1 mapa artificial
# 2 - Gerar fundos a partir de 5 mapas artificiais
# 3 - Gerar fundos a partir de 10 mapas artificiais


# print('Quantos fundos deseja gerar?')
# fundos = input()
# print('Gerar os fundos a partir de quantos mapas artificiais')
# mapas = input()

# from easygui import *
# text = "Selected any one item"
# title = "Window Title GfG"
# choices = ["Geek", "Super Geeek", "Super Geek 2", "Super Geek God"] 
# output = choicebox(text, title, choices) 
# title = "Message Box"
# message = "You selected : " + str(output) 
# msg = msgbox(message, title)

print('Quantos fundos deseja gerar?')
fundos = input()

print('Selecione uma das opções abaixo:')
print('1 - Gerar fundos a partir de 1 mapa artificial')
print('2 - Gerar fundos a partir de 5 mapas artificiais')
print('3 - Gerar fundos a partir de 10 mapas artificiais')
