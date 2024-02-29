import sys,pickle

# Linux path
# sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

# Windows path
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

import numpy as np
import matplotlib.pyplot as plt
import json
from Slice_mapper import slice_mapper as slice_mapper
from PIL import Image

from Utils import functions


# reads a file and returns an array of the file data
# The array is in column/row format. The code that performs the extraction writes in column/row format
# Several parts of vessel analysis and slice_mapper use this format.
def return_paths(json_file):
    """Function that reads a JSON file and returns paths 1 and 2 for one or multiple manual blood vessel markings.

    Parameters:
    -----------
    json_file: str
        File containing the coordinates, rows, and columns of the blood vessel location with a .json extension.
    Returns:
    -----------
    array_paths: list, containing ndarray
        Returns path1 and path2 for one or multiple extracted vessels.
        The values stored in path1 and path2 are manual markings made on the vessels.
    """
    # Read JSON file
    data = json.load(open(json_file, 'r'))

    # Convert all items read into np.array
    array_paths = [np.array(item) for item in data]

    # Function with one line to reverse all values
    # path1 = [np.array(item)[:,::-1] for item in data]
    return array_paths


def set_range(array_1, array_2, extra_range=6):
    """Function that takes the first two values of the rows and columns of the vectors passed as parameters and
       returns the range. It calculates the Euclidean distance between the four points.

    Parameters:
    -----------
    array_1: ndarray, float
        Vector for path 1
    array_2: ndarray, float
        Vector for path 2
    Returns:
    -----------
    range_value: int
        Returns the integer value of the Euclidean distance calculation.
        It is used to delimit the region that will be displayed in the image.
    """
    # Get column 1 of vector 1
    column1 = array_1[0][0]

    # Get row 1 of vector 1
    row1 = array_1[0][1]

    # Get column 2 of vector 2
    column2 = array_2[0][0]

    # Get row 2 of vector 2
    row2 = array_2[0][1]

    # Range is the square root of the result of the quadratic difference between the two points of the two vectors - Euclidean distance
    # The extra_range variable allows the region set by the range to be slightly larger. The range is increased both above and below the mapped values
    range_value = int(np.sqrt((row1 - row2) ** 2 + (column1 - column2) ** 2) + extra_range)

    return range_value


def return_rows_columns(paths):
    """Function that returns the minimum and maximum values of two vectors.

    Parameters:
    -----------
    paths: list, containing ndarray
        List containing two ndarray vectors (path1 and path2)
    Returns:
    -----------
    min_row: int
        Minimum value found among the rows
    min_column: int
        Minimum value found among the columns
    max_row: int
        Maximum value found among the rows
    max_column: int
        Maximum value found among the columns
    """
    # Get the first position of the vector
    path1 = paths[0]

    # Get the second position of the vector
    path2 = paths[1]

    # Calculate the minimum values for columns and rows for each path
    min_column1, min_row1 = np.min(path1, axis=0)
    min_column2, min_row2 = np.min(path2, axis=0)

    # Calculate the maximum values for columns and rows for each path
    max_column1, max_row1 = np.max(path1, axis=0)
    max_column2, max_row2 = np.max(path2, axis=0)

    # Determine the overall minimum values for columns and rows
    min_column = int(np.min([min_column1, min_column2]))
    min_row = int(np.min([min_row1, min_row2]))

    # Determine the overall maximum values for columns and rows
    max_column = int(np.max([max_column1, max_column2]))
    max_row = int(np.max([max_row1, max_row2]))


    return min_row, min_column, max_row, max_column


def resize_image(paths, image_path):
    """Function that takes a vector containing paths 1, 2 and the address of an image, resizing its dimensions to
     those set by the variables containing the smallest and largest values of the rows and columns contained in paths 1 and 2.

    Parameters:
    -----------
    paths: list, containing ndarray
        List containing two ndarray vectors (path1 and path2)
    image_path: str
        Image address that displays its location
    Returns:
    -----------
    img1: ndarray, image
        Resized image
    translated_paths: list, float
        List containing a pair of vectors that have been translated from the first point of the smallest row
        and column, minus padding, to fit in the resized image
    first_point: ndarray
        Contains information from the smallest row and column, minus padding of the original image
    """
    # Padding set to show a region slightly larger than the vessel in question
    padding = 5

    # Returns the smallest and largest rows and columns of the paths
    min_row, min_column, max_row, max_column = return_rows_columns(paths)

    # Get the first_point at the position of the smallest column and the smallest row, decreased by padding
    first_point = np.array([min_column - padding, min_row - padding])

    # Absorb the values of the paths decreased from the first point, sweeping both vectors
    translated_paths = [path - first_point for path in paths]

    # Image that absorbs img_path and shows the region delimited by the parameters by the smallest and largest rows/columns
    # The "-padding" parameter allows a slightly larger region to be taken in all values of rows/columns
    img1 = np.array(Image.open(image_path))[min_row - padding:max_row + padding,
           min_column - padding:max_column + padding]

    return img1, translated_paths, first_point


def generate_vessel_cross(img, trans_paths_0, trans_paths_1, range_value, delta_eval=1., smoothing=0.01):
    """Function that creates the vessel model and transversal paths.

    Parameters:
    -----------
    img: ndarray, float
        Resized image containing the area of the extracted vessel.
    trans_paths_0: ndarray, float
        Translated paths at position 0 of the translated paths vector.
    trans_paths_1: ndarray, float
        Translated paths at position 1 of the translated paths vector.
    range_value: int
        Variable that defines how much upper and lower limit the image will have, directly related to the number of lines in the created map.
    delta_eval: float
        Variable that defines the intervals.
    smoothing: float
        Variable that defines the degree of smoothing.
    Returns:
    -----------
    vessel_model: object VesselModel
        Returns the vessel model with an instantiated object of the VesselModel class.
    cross_paths: ndarray, float
        Transversal paths.
    """
    vessel_model, cross_paths = slice_mapper.map_slices(img, trans_paths_0, trans_paths_1, delta_eval, smoothing, range_value)

    return vessel_model, cross_paths

def plot_figure(img, vessel_model, cross_paths,i):
    """Function that creates the vessel model and transversal paths.

    Parameters:
    -----------
    img: ndarray, float
        Resized image containing the area of the extracted vessel.
    vessel_model: object VesselModel
        Returns the vessel model with an instantiated object of the VesselModel class.
    cross_paths: ndarray, float
        Transversal paths.
    Returns:
    -----------
        Plots the resized image, along with the vessel model, transversal paths, and translated paths 1 and 2,
        in three different ways:
        1 - with the mapped values having the minimum at 0 and maximum at 60
        2 - values mapped in the standard range, from 0 to 255
        3 - values mapped between the minimum 0 and maximum in the values found in the mapping
    """
    vessel_map = vessel_model.vessel_map

    # Plotting using slice_mapper.plot_model
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot()
    slice_mapper.plot_model(img, vessel_model, cross_paths, ax)   
    norm = ax.images[0].norm
    norm.vmin, norm.vmax = 0, 60
    plt.axis('off')    
    plt.savefig(f'img{i}.svg',format='svg')

    # Plotting the mapped values with specified vmin and vmax
    plt.figure(figsize=[12, 10])
    plt.title("Vmin=0 and Vmax=60")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')
    plt.axis('off')
    #plt.savefig('img2.svg',format='svg')

    # Plotting the mapped values reversed with vmin and vmax
    plt.figure(figsize=[12, 10])
    plt.title("Vmin=0 and Vmax=255")
    plt.plot()
    plt.imshow(vessel_map.mapped_values[::-1], 'gray', vmin=0, vmax=255)
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')
    plt.axis('off')
    #plt.savefig('img3.svg',format='svg')



    # Plotting the mapped values with dynamic vmax (max mapped value)
    plt.figure(figsize=[12, 10])
    plt.title("Vmin=0 and Vmax=max mapped value")
    plt.plot()
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=vessel_map.mapped_values.max())
    plt.plot(vessel_map.path1_mapped, c='green')
    plt.plot(vessel_map.path2_mapped, c='green')
    plt.axis('off')
    #plt.savefig('img4.svg',format='svg')

def generate_vessel_models(params):
    '''Function that reads all .json files and transforms them into a vessel model 
       by writing to a folder designated as the output directory
    
    Parameters:
    -----------
    params['dir_json']: text
        Directory of jscon vectors 
    params['dir_images']: text
        Directory of original images
    params['out_dir_save_data']: text
        Output directory of save data 

    Returns:
    -----------
        Plots the vessel models and maps created and also saves the vessel models
        in .pickle files within the defined output directory    
    '''
    root_json = params['dir_json']
    root_imgs = params['dir_images']
    root_out = params['out_dir_save_data']
    
    json_array = functions.read_directories(root_json,exclude_json='yes')

    for i in range(len(json_array)):

        path = f"{json_array[i].replace('.json','')}"

        file_path = f'{root_json}/{path}.json'

        image_path = f'{root_imgs}/{path}.tiff'

        # retrieve the file and store it in an array
        array_path = return_paths(file_path)

        # read the image
        img = np.array(Image.open(image_path))

        # get the integer half of the vector
        half_array = len(array_path) // 2

        x = 0
        for j in range(half_array):
            img, translated_paths, first_point = resize_image(array_path[x:x+2], image_path)
            range_value = set_range(array_path[0], array_path[1])
            vessel_model, cross_section = generate_vessel_cross(img, translated_paths[0], translated_paths[1], range_value)
            rand = np.random.randint(0,1000)
            plot_figure(img, vessel_model, cross_section,rand)

            # section to save the .pickle file
            #data_dump = {"img_file": image_path, "vessel_model": vessel_model, "primeiro_ponto": first_point}
            #save_data = f'{root_out}/{path}_savedata{i}.pickle'
            #pickle.dump(data_dump, open(save_data, "wb"))
            x += 2

    