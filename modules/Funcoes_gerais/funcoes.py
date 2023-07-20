import os
import sys
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

def ler_diretorios(dir):
    filenames = []
    for filename in os.listdir(dir):
    #     # Use only images having magnification 20x
        if 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@40-Image 3-20X' in filename:
    #     if 'T-3 Weeks' in filename:      
        #filenames.append(filename.split('.')[0])
            filenames.append(filename)
    #filenames = filenames[:20]
    return filenames