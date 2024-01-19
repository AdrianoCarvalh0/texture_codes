import os, sys, json
import sys
import time
import tracemalloc
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

def read_directories(directory, img=None):
    # Get a list of filenames in the specified directory
    filenames = []
    for filename in os.listdir(directory):
        if img is not None:
            # If 'img' is provided, filter filenames containing it
            if img in filename:   
                filenames.append(filename)
        else:
            filenames.append(filename)    
    return filenames

def write_array_to_file(array_list, filename):  
    # Convert array items to lists before writing to JSON file
    list_of_lists = [item.tolist() for item in array_list]
    json.dump(list_of_lists, open(filename, 'w'), indent=2)

def write_dict_to_file(dictionary, filename):
    array_list = dictionary["curve"]
    distance = dictionary["distance"]  
    # Convert array items to lists before writing to JSON file
    list_of_lists = [item.tolist() for item in array_list]  
    json.dump(list_of_lists, open(filename, 'w'), indent=2)

def measure_time_memory(description, function):
    # Start tracing memory usage
    tracemalloc.start()
    start_time = time.time()
    
    # Execute the specified function
    function()
    
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    execution_time = end_time - start_time

    print(f"{description} took {execution_time} seconds, and the peak memory usage was {peak_memory/1024**3} GBs.")
    tracemalloc.stop()
