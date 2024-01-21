from pathlib import Path
import sys
import numpy as np
import pickle
import vessel_statistics as vs

# linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

# windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

if __name__ == '__main__':

    # Directory containing vessel model pickle files
    dir = f'{root_dir}/Vessel_Models_pickle'
   
    # Retrieve file names and the number of items in the directory
    nom, tam = vs.ready_directory(dir)

    maximum = []
    minimum = []
    vetor_diametros = []

    # Iterate over all items in the directory
    for i in range(1):  
        # Get the file name and store it in the local variable
        local = nom[i]
        # Load the .pickle file
        data_dump = pickle.load(open(local, "rb"))
        # Extract the three indices from the .pickle file
        vessel_model = data_dump['vessel_model']
        first_point = data_dump['primeiro_ponto']
        img_file = data_dump['img_file']  

        # Instantiate the vessel_map variable
        vessel_map = vessel_model.vessel_map 

        # Get the half integer size of vessel_map.mapped_values
        half_size_vessel_map = len(vessel_map.mapped_values) // 2

        # Variables that get the index of the minimum and maximum values of the middle position
        max_index = np.argmax(vessel_map.mapped_values[half_size_vessel_map])
        min_index = np.argmin(vessel_map.mapped_values[half_size_vessel_map])  

        # Variables that get the values stored in the minimum and maximum values found
        maximum.append(vessel_map.mapped_values[half_size_vessel_map][max_index])
        minimum.append(vessel_map.mapped_values[half_size_vessel_map][min_index])

        # Returns the mean of all mapped values along the rows
        means = np.mean(vessel_map.mapped_values, axis=1)

        # Returns the standard deviation of all mapped values along the rows
        std_dev = np.std(vessel_map.mapped_values, axis=1)  

        # Returns the standard deviation of all mapped values along the columns
        std_dev2 = np.std(vessel_map.mapped_values, axis=0)    

        # Plot of the vessel map, min value is 0, and max value is 60
        vs.plot_vessel_map(vessel_map) 

        # Plot of the clipping
        vs.plot_clipping(vessel_map)
        
        # Plot of the intensity lines, one above and one below
        vs.plot_intensity_lines(vessel_map, half_size_vessel_map) 

        # Plot showing the difference between the mean and standard deviation
        vs.plot_fill_means_std_dev(means, std_dev)

        #testando
        intensity_cols_values_all, _, vector_diameter = vs.return_all_instisitys_normal(vessel_map)

        # Attempt to plot the difference between the standard deviation with the normalized column
        vs.plot_fill_means_std_dev_normal_all(intensity_cols_values_all)
        
        # Plot the diameter of the vessel
        vs.plot_diameter_vessel(vessel_map)  

        # Plot the intensity of the columns, shows where the vessel starts and ends
        vs.plot_intensity_cols_with_line_vessel(vessel_map)

        # Plot the intensity of the columns normalized with the centerline (half of the column indices), 
        # shows where the vessel starts and ends, removing the dependence on starting from the point (0,0)
        #Está dando erro - VER
        #vs.plot_intensity_cols_with_line_vessel_normal(vessel_map)

        # Plot all normalized intensities
        intensities_common_axis, l2_chapeu_axis, vector_diameter = vs.return_all_instisitys_normal(vessel_map)

        vs.plot_all_intensities_columns(intensities_common_axis, l2_chapeu_axis)

        vs.plot_fill_means_std_dev_normal_all(intensities_common_axis)

        # Plot the difference between the maximum and minimum of all extractions
        vs.plot_min_max_medial_line(minimum, maximum)

        # Use plt.xaxis.set_Visible(False)

        # Use Inkscape - image editing software
        diameter = np.abs(vessel_map.path1_mapped - vessel_map.path2_mapped)
        a = np.array(diameter)
        media = np.mean(a)
        vetor_diametros.append(media)
