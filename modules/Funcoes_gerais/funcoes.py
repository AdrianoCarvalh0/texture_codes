import os
import sys
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

def ler_diretorios(dir, img=None):
    filenames = []
    for filename in os.listdir(dir):
        if img is not None:
            if img in filename:   
                filenames.append(filename)
        else:
            filenames.append(filename)
    
    return filenames