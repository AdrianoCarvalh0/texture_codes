import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def ready_directory(directory):
    """Function that reads all files in a directory, returning the quantity and names of the files.

    Parameters:
    -----------
    directory: str
        Name of the location where the directory to be read is located.
    Returns:
    -----------
    names: list, str
        List of file names being read in the set directory.
    quantity: int
        Number of files in the directory.
    """

    file_quantity = 0
    names_list = []
    # Scans files and adds names to the 'names' variable and quantity to the 'quantity' variable.
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            names_list.append(path)
            file_quantity += 1   
    return names_list, file_quantity

def plot_vessel_map(vessel_map):
    """Function that plots the vessel map. Maps values from zero as the minimum to 60 as the maximum.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.

    Returns:
    -----------
        Plots the intensity values of the blood vessel pixels.
    """
    plt.figure(figsize=[12, 10])   

    # 'mapped_values' are the intensity values of the blood vessel pixels.
    plt.imshow(vessel_map.mapped_values, 'gray', vmin=0, vmax=60)

    # Shows the values of path1 mapped in yellow.
    plt.plot(vessel_map.path1_mapped, c='yellow')

    # Shows the values of path2 mapped in yellow.
    plt.plot(vessel_map.path2_mapped, c='yellow')        
    plt.show()


def plot_intensity_lines(vessel_map, half_size_vessel_map):
    """Function that plots the intensity of the median line, one above and one below the mapped values.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    half_size_vessel_map: int
        Integer half of the division of the size of the mapped values by 2.
    Returns:
    -----------
        Plot of the intensity of the median line, one above and one below the mapped values.
    """
    plt.figure(figsize=[12, 10])    
    plt.title(f'Intensities of the medial line in lines {half_size_vessel_map - 1}, {half_size_vessel_map}, and {half_size_vessel_map + 1}')

    # above
    plt.plot(vessel_map.mapped_values[half_size_vessel_map - 1].flatten(),
             label=f'Position:  {half_size_vessel_map - 1}')
    # centerline
    plt.plot(vessel_map.mapped_values[half_size_vessel_map].flatten(), label=f'Position:  {half_size_vessel_map}')

    # below
    plt.plot(vessel_map.mapped_values[half_size_vessel_map + 1].flatten(), label=f'Position:  {half_size_vessel_map + 1}')

    plt.legend(loc='lower right')
    plt.xlabel('Positions')
    plt.ylabel('Intensities')   
    plt.show() 


def plot_fill_means_std_dev(means, std_dev):
    """Function that plots the difference between the mean and standard deviation.

    Parameters:
    -----------
    means: ndarray, float
        Mean of all mapped values along the lines.
    std_dev: ndarray, float
        Standard deviation of all mapped values along the lines.
    Returns:
    -----------
        Plots the difference between the mean and standard deviation.
    """

    plt.figure(figsize=[12, 10])    
    plt.title("Filling between the mean intensity and standard deviation along the lines")

    # shows the shading
    plt.fill_between(range(len(means)), means - std_dev, means + std_dev, alpha=0.3)

    # shows the mean
    plt.plot(range(len(means)), means)   
    plt.show()


def plot_diameter_vessel(vessel_map):
    """Function that plots the diameter of the mapped vessels.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    Returns:
    -----------
        Plots the diameter of the mapped vessels.
    """      

    # diameter is the absolute difference between the two mapped paths.
    diameter = np.abs(vessel_map.path1_mapped - vessel_map.path2_mapped)    
  
    plt.figure(figsize=[12, 10])
    plt.title("Diameter of the vessel")    
    plt.xlabel('Column Index')
    plt.ylabel('Diameter')

    # diameter is a float, so it required range(len)
    plt.plot(range(len(diameter)), diameter)    
    plt.show()  


def plot_all_diameter(array):
    """Function that plots the diameter of all vessels.

    Parameters:
    -----------
    array: array
        vector that contains all Diameter of all vessels
    Returns:
    -----------    
        Plots the diameter of all vessels.
    """        
    
    plt.figure(figsize=[12, 10])
    plt.title("Diameter of all vessels")    
    plt.xlabel('Files')
    plt.ylabel('Diameter')

    # diameter is a float, so it required range(len)
    plt.plot(range(len(array)),array)    
    plt.show() 

def return_intensity_cols(vessel_map):
    """Function that stores all column intensities.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    Returns:
    -----------
     intensity_cols_values_all: list, ndarray
        List containing all column intensity values in numpy array format.
    """

    # number of rows and columns in the vessel map
    num_rows, num_cols = vessel_map.mapped_values.shape

    intensity_cols_values_all = []

    # stores all column intensities along the rows
    for i in range(num_cols):
        intensity_cols_values_all.append(vessel_map.mapped_values[0:num_rows, i])
    return intensity_cols_values_all


def return_clipping(vessel_map):
    """Function that clips an image.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    Returns:
    -----------
    clipping: ndarray, float
        Clipped image showing the area where the vessel is located. This image displays only the mapped values
        with a padding of 1 pixel only.
    """
    padding = 1
    # minimum line of path2
    line_min_path2 = int(np.min(vessel_map.path2_mapped) + padding)
    # maximum line of path1
    line_max_path1 = int(np.max(vessel_map.path1_mapped) + padding)

    # all mapped values
    img_path = vessel_map.mapped_values

    # fetching the number of columns in the image
    _, num_cols = img_path.shape

    # the clipping is done from the minimum line to the maximum line, and from columns ranging from 0 to the number of existing columns
    clipping = (img_path[line_min_path2:line_max_path1, 0:num_cols])
    return clipping


def plot_clipping(vessel_map):
    """Function that plots an image with minimum values of zero and maximum of 60.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    Returns:
    -----------
        Plots the clipped image showing the area where the vessel is located with one pixel of padding.
    """
    # calls the function that returns the clipping
    clipp = return_clipping(vessel_map)

    plt.figure(figsize=[12, 10])
    plt.title("Image clipping")
    plt.imshow(clipp, 'gray', vmin=0, vmax=60)    
    plt.show()



def plot_intensity_cols_with_line_vessel(vessel_map):
    """Function that plots column intensities. Also displays where the vessel starts and ends
    through central bars, perpendicular to the y-axis.

    Parameters:
    -----------
    vessel_map: object VesselMap
       Instance of the VesselMap object.
    Returns:
    -----------
        Plots column intensities and displays vessel boundaries on the left and right.
    """
    array_min_path = []
    array_max_path = []

    # number of rows and columns of the mapped values
    num_rows, num_cols = vessel_map.mapped_values.shape
    
    # various colors to align the color of the columns to be displayed with the v_lines that show the vessel boundaries
    colors = ['blue', 'green', 'red', 'orange', 'gray']

    # calls the function that gets all column intensities
    intensity_cols_values_all = return_intensity_cols(vessel_map)

    # Getting position 0, 1/4, 1/2, 3/4, and end of the columns
    demarcated_columns = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

    plt.figure(figsize=[12, 10])
    plt.title(
        f'Column intensities {demarcated_columns[0]}, {demarcated_columns[1]}, {demarcated_columns[2]}, {demarcated_columns[3]} and {demarcated_columns[4]}')
    plt.xlabel('Row Index')
    plt.ylabel('Intensity')
    for i in range(len(demarcated_columns)):
        # plots the positions existing in the marked columns in the vector containing all column intensities
        plt.plot(range(num_rows), intensity_cols_values_all[demarcated_columns[i]],
                 label=f'Position:  {demarcated_columns[i]}', color=colors[i])
    plt.legend(loc='lower right')

    liv_list_vlines = []
    lfv_list_vlines = []
    for j in range(len(demarcated_columns)):
        min_path = np.argmin(intensity_cols_values_all[demarcated_columns[j]])
        array_min_path.append(intensity_cols_values_all[demarcated_columns[j]][min_path])

        max_path = np.argmax(intensity_cols_values_all[demarcated_columns[j]])
        array_max_path.append(intensity_cols_values_all[demarcated_columns[j]][max_path])

        liv_list_vlines.append(vessel_map.path1_mapped[demarcated_columns[j]])
        lfv_list_vlines.append(vessel_map.path2_mapped[demarcated_columns[j]])
       
    plt.vlines(liv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors, ls='--')
    plt.vlines(lfv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors, ls='--')    
    plt.show()    


def plot_intensity_cols_with_line_vessel_normal(vessel_map, demarcated_columns=None):
    """Function that plots column intensities. Also displays where the vessel starts and ends
    through central bars, perpendicular to the y-axis. Here we will show some specific columns along the
    vessel. The column intensities will be kept, but the axis will be normalized according to the center line.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    demarcated_columns: NoneType
        The field is None by default, being set later to get 5 columns along the vessel. If this
        parameter is filled, the columns will be the ones passed as parameters
    Returns:
    -----------
        Plots column intensities and displays vessel boundaries on the left and right.
    """
    num_rows, num_cols = vessel_map.mapped_values.shape   

    if (demarcated_columns is None):
        # Showing position 0, 1/4, 1/2, 3/4, and end of the columns
        demarcated_columns = [0, (num_cols // 4), (num_cols // 2), ((num_cols * 3) // 4), (num_cols - 1)]

    # receives a vector of colors
    colors = ['blue', 'green', 'red', 'orange', 'gray']

    # fetches all intensities of all columns
    intensity_cols_values_all = return_intensity_cols(vessel_map)

    # Integer remainder of the number of lines divided by 2
    half_rows = num_rows // 2

    # vector created to store the positions
    number_rows_vector = []
    for i in range(num_rows):
        # creating a vector of size 27 positions
        number_rows_vector.append(i)

    l_hat = []
    for j in range(len(number_rows_vector)):
        # Formula (L1'' = 2L'/(Lfv1-Liv1))
        l_hat.append(2 * (number_rows_vector[j] - half_rows))

    lfv_list = []
    liv_list = []
    vector_diameter = []
    l2_hat_all = []
    for col in demarcated_columns:
        lfv = vessel_map.path2_mapped[col]
        liv = vessel_map.path1_mapped[col]
        lfv_list.append(lfv)
        liv_list.append(liv)
        # gets the last value that was added to the list
        vector_diameter.append(abs(lfv - liv))

        l2_hat = []
        for k in range(len(l_hat)):
            # Formula (L2'' = 2L'/(Lfv-Liv))
            l2_hat.append(2 * l_hat[k] / vector_diameter[-1])
        l2_hat_all.append(l2_hat)

    plt.figure(figsize=[12, 10])
    for i in range(len(demarcated_columns)):      
        plt.plot(l2_hat_all[i], intensity_cols_values_all[demarcated_columns[i]], 
                 label=f'Position:  {demarcated_columns[i]}', color=colors[i])
    plt.legend(loc='lower right')

    liv_list_vlines = []
    lfv_list_vlines = []
    # l = (number_rows_vector - half_rows) / diameter
    for k in range(len(demarcated_columns)):
        formula1 = 2 * (liv_list[k] - half_rows) / vector_diameter[k]
        formula2 = 2 * (lfv_list[k] - half_rows) / vector_diameter[k]
        liv_list_vlines.append(formula1)
        lfv_list_vlines.append(formula2)

    array_min_path = []
    array_max_path = []

    for i in range(len(demarcated_columns)):
        min_path = np.argmin(intensity_cols_values_all[demarcated_columns[i]])
        array_min_path.append(intensity_cols_values_all[demarcated_columns[i]][min_path])

        max_path = np.argmax(intensity_cols_values_all[demarcated_columns[i]])
        array_max_path.append(intensity_cols_values_all[demarcated_columns[i]][max_path])
    plt.vlines(liv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors, ls='--')
    plt.vlines(lfv_list_vlines, np.min(array_min_path), np.max(array_max_path), color=colors, ls='--')

    # SEE
    plt.xlabel('Positions')
    plt.ylabel('Intensity')

    plt.legend(loc='lower right')    
    plt.show()


def return_all_instisitys_normal(vessel_map):
    """Function that returns all intensities normalized with the central line.

    Parameters:
    -----------
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    Returns:
    -----------
    intensities_common_axis: ndarray, float
        Vector containing normalized intensities.
    l2_hat_axis: ndarray, float
        Contains information about where the axis should start and end. There is a modification in the graph display,
        instead of starting from the origin [0,0]. It (the origin) will start depending on the number of rows that exist.
    """

    num_rows, num_cols = vessel_map.mapped_values.shape

    # Fetches all intensities of all columns
    intensity_cols_values_all = return_intensity_cols(vessel_map)
    
    # Integer remainder of the number of lines divided by 2
    half_rows = num_rows // 2

    # Vector created to store the positions
    number_rows_vector = []
    for i in range(num_rows):
        # Creating a vector of size N positions
        number_rows_vector.append(i)

    l = []
    for j in range(len(number_rows_vector)):
        # In this loop, I add to the vector created earlier. Putting the lines divided by 2 ==> lc = num_rows//2
        l.append(number_rows_vector[j] - half_rows)

    lfv_list = []
    liv_list = []
    vector_diameter = []

    l_all = []
    for col in range(len(intensity_cols_values_all)):
        liv = vessel_map.path1_mapped[col]
        lfv = vessel_map.path2_mapped[col]
        liv_list.append(liv)
        lfv_list.append(lfv)
        # Gets the last value that was added to the list
        vector_diameter.append(abs(lfv - liv))

        l2 = []
        for k in range(len(l)):
            # Formula (L1'' = 2L'/(Lfv1-Liv1))
            l2.append(2 * l[k] / vector_diameter[-1])
        l_all.append(l2)

    l2_min, l2_max = np.min(l_all), np.max(l_all)

    l2_hat_axis = np.linspace(l2_min, l2_max, num_rows)

    # Create interpolating functions
    l2_hat_funcs = []
    for l2, intens in zip(l_all, intensity_cols_values_all):
        l2_hat_func = interp1d(l2, intens, kind='linear', bounds_error=False)
        l2_hat_funcs.append(l2_hat_func)

    # Calculate intensities for each point on the common axis
    intensities_common_axis = np.zeros((len(l2_hat_funcs), len(l2_hat_axis)))
    for col, l2_val in enumerate(l2_hat_axis):
        for row, l2_hat_func in enumerate(l2_hat_funcs):
            intensities_common_axis[row, col] = l2_hat_func(l2_val)

    return intensities_common_axis, l2_hat_axis


def plot_all_intensities_columns(intensities_common_axis, l2_hat_axis):
    """Function that plots all intensities normalized from the central line.

    Parameters:
    -----------
    intensities_common_axis: ndarray, float
        Vector containing normalized intensities.
    l2_hat_axis: ndarray, float
        Contains information about where the axis should start and end. There is a modification in the graph display,
        instead of starting from the origin [0,0]. It (the origin) will start depending on the number of rows that exist.
    Returns:
    -----------
        Plots all column intensities.
    """

    plt.figure(figsize=[12, 10])
    for intens in intensities_common_axis:
        plt.plot(l2_hat_axis, intens)
    #plt.savefig('plot_all_intensities_columns.pdf')
    plt.show()


# Function that plots the minimum and maximum of the medial line of all extractions
def plot_min_max_medial_line(minimum, maximum):
    """Function that plots all minimum and maximum values of the medial line for each extracted vessel. Each vessel model
    has a maximum and minimum intensity value of the medial line. This function is used to visualize
    these variations.

    Parameters:
    -----------
    minimum: list, float
        List containing the minimum intensity values of the medial line for each vessel.
    maximum: list, float
        List containing the maximum intensity values of the medial line for each vessel.
    Returns:
    -----------
       Plots all minimum and maximum values of the medial line for each extracted vessel.
    """

    maximum = np.array(maximum)
    minimum = np.array(minimum)
    plt.figure(figsize=[12, 10])
    plt.title(f'Maximum and minimum of the medial line:')
    plt.ylabel('Number')
    plt.xlabel('Values')
    plt.plot(minimum.flatten(), label=f'Minimum')
    plt.plot(maximum.flatten(), label=f'Maximum')
    plt.legend(loc='lower right')
    #plt.savefig('plot_min_max_medial_line.pdf')
    plt.show()
