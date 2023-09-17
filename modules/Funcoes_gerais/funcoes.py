import os, sys, json
import sys
import time
import tracemalloc
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


def gravar_array_arquivo(array_list, filename):  
  lista2 = [item.tolist() for item in array_list]
  json.dump(lista2, open(filename, 'w'), indent=2)

def gravar_dict_arquivo(dict, filename):
  array_list = dict["curve"]
  distancia = dict["distancia"]  
  lista2 = [item.tolist() for item in array_list]  
  json.dump(lista2, open(filename, 'w'), indent=2)


def calcular_tempo_memoria(str, funcao):
    tracemalloc.start()
    start_time = time.time()          
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    execution_time = end_time - start_time

    print(f"Criação da {str} {execution_time} seconds, and the peak memory usage was {peak_memory/1024**3} GBs.")
    tracemalloc.stop()  