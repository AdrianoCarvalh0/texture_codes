from pathlib import Path
import sys

# linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

# path windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

from Slice_mapper import slice_mapper
import numpy as np
import pickle
from PIL import Image
import vessel_analysis as va

if __name__ == '__main__':

    image = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@64-Image 4-20X'

    file_path = f'{root_dir}/Extracted_json_vectors/novos/{image}.json'

    image_path = f'{root_dir}/Images/vessel_data/images/{image}.tiff'

    # retrieve the file and store it in an array
    array_path = va.return_paths(file_path)

    # read the image
    img = np.array(Image.open(image_path))

    # get the integer half of the vector
    half_array = len(array_path) // 2

    x = 0
    for i in range(half_array):
        img, translated_paths, first_point = va.resize_image(array_path[x:x+2], image_path)
        range_value = va.set_range(array_path[0], array_path[1])
        vessel_model, cross_section = va.generate_vessel_cross(img, translated_paths[0], translated_paths[1], range_value)
        va.plot_figure(img, vessel_model, cross_section)
        va.plot_figure2(img, vessel_model, cross_section)

        # section to save the .pickle file
        data_dump = {"img_file": image_path, "vessel_model": vessel_model, "first_point": first_point}
        save_data = f'{root_dir}/Vessel_models_pickle/novos/{image}_savedata{i}.pickle'
        pickle.dump(data_dump, open(save_data, "wb"))
        x += 2
