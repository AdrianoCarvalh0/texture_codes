import sys, scipy, pickle, json
import skimage as ski
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

# Linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

# Windows
sys.path.insert(0, r"C:\Users\adria\Documents\Mestrado\texture_codes\modules\Slice_mapper")

from shapely.geometry import Point, LineString
from PIL import Image
from scipy import ndimage


def return_paths(json_file):
    """Function that reads a JSON file and returns paths 1 and 2 from one or multiple manual blood vessel markings.

    Parameters:
    -----------
    json_file: str
        File containing coordinates, rows, and columns of the blood vessel location with a .json extension
    Returns:
    -----------
    array_paths: list, containing ndarray
        Returns path1 and path2 of one or multiple extracted vessels.
        The values stored in path1 and path2 are manual markings made on the vessels.
    """
    # Read the JSON file
    q = json.load(open(json_file, 'r'))

    # Convert all items read into np.array
    array_paths = [np.array(item) for item in q]

    # Function with one line to invert all values
    # path1 = [np.array(item)[:,::-1] for item in q]
    return array_paths

def find_most_frequent_pixel(map):
    image = Image.fromarray(map)

    # Convert the image to grayscale
    image_gray = image.convert("L")

    # Get the histogram of pixel values
    histogram = image_gray.histogram()

    # Create a list of tuples (pixel value, frequency)
    pixel_freq_pairs = list(enumerate(histogram))

    # Sort the list in descending order of frequency
    sorted_pixel_freq_pairs = sorted(pixel_freq_pairs, key=lambda x: x[1], reverse=True)

    # Separate pixel values and frequencies
    pixels, freqs = zip(*sorted_pixel_freq_pairs)

    # Find the pixel value with the highest frequency
    most_frequent_pixel = pixels[0]

    return most_frequent_pixel

def plot_points(x, y, title=None):  
    idx = range(len(x))
    fig, ax = plt.subplots()
    plt.imshow(np.zeros((32, 52)), 'binary')
    ax.scatter(x, y)
    for i, txt in enumerate(idx):
        ax.annotate(txt, (x[i], y[i]))
    if title:
        plt.title(title)

def transform_v2(src, dst, img, order=0):
    """Transform image."""
    
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    
    if img.ndim == 2:
        # Add channel dimension if 2D
        img = img[..., None]
    num_rows, num_cols, num_channels = img.shape
    
    # Find minimum and maximum values for dst points
    min_dst_col, min_dst_row = dst.min(axis=0).astype(int)
    max_dst_col, max_dst_row = np.ceil(dst.max(axis=0)).astype(int)
    ul_point_dst = np.array([min_dst_col, min_dst_row])
    # New origin point for the space containing both src and dst
    new_origin = np.minimum(ul_point_dst, np.zeros(2, dtype=int))
    translation = np.abs(new_origin)

    src -= new_origin
    dst -= new_origin

    # Create a new source image considering the new origin
    img_proper = np.zeros((num_rows + translation[1], num_cols + translation[0], num_channels), dtype=img.dtype)
    img_proper[translation[1]:translation[1] + num_rows, translation[0]:translation[0] + num_cols] = img
    output_shape = (max_dst_row - new_origin[1], max_dst_col - new_origin[0])

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    img_out = warp(img_proper, tform.inverse, output_shape=output_shape, order=order)
    
    if img.ndim == 2:
        img_out = img_out[0]
        
    return img_proper, img_out, src, dst, tform, translation, new_origin


def plot(img_proper, img_out, src, dst, vmax):

    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.imshow(img_proper, 'gray', vmin=0, vmax=vmax)
    plt.plot(src[:, 0], src[:, 1], 'o')
    plt.title('Adjusted source image')
    plt.axis((0, img_proper.shape[1], img_proper.shape[0], 0))
    plt.subplot(1, 2, 2)
    plt.imshow(img_out, 'gray', vmin=0, vmax=vmax)
    plt.plot(dst[:, 0], dst[:, 1], 'o')
    plt.axis((0, img_out.shape[1], img_out.shape[0], 0))

def returns_largest_line(line1, line2, line3):
    len_line1 = len(line1.coords)
    len_line2 = len(line2.coords)
    len_line3 = len(line3.coords)
    x = np.array([len_line1, len_line2, len_line3])
    max_len_line = np.max(x)              
    return max_len_line

def returns_new_points_from_lines(distance, line):  
    points = []
    vector = []
    for dist in distance:
        p = line.interpolate(dist, normalized=True)
        points.append(p) 
    for p in points:
        vector.append([p.x, p.y])  
    return vector

def returns_binary_image_without_artifacts(vessel_map, img_bin):

    num_rows, num_cols = img_bin.shape

    min_line = (np.min(np.rint(vessel_map.path2_mapped)) - 1)
    max_line = (np.max(np.rint(vessel_map.path1_mapped)) + 1)   
    min_line_int = int(min_line)   
    max_line_int = int(max_line)

    binary_image_without_artifacts = img_bin.copy().astype('int32')

    for num_row in range(min_line_int):
        for num_col in range(num_cols):
            binary_image_without_artifacts[num_row, num_col] = 0  

    for i in range(max_line_int, num_rows):
        for num_col in range(num_cols):
            binary_image_without_artifacts[i, num_col] = 0      
    return binary_image_without_artifacts


def estimate_background(image: np.ndarray, label: np.ndarray, window_size: int=15) -> np.ndarray:
    """Function that creates an artificial background."""
    
    contains_foreground_pixels = lambda target_label: np.count_nonzero(target_label) > 0

    # Divide the image into patches
    list_slices = []
    window_size = 15
    h_window_size = window_size // 2
    i = h_window_size
    j = h_window_size

    while i < image.shape[0] - h_window_size:
        j = h_window_size
        while j < image.shape[1] - h_window_size:
            list_slices.append((slice(i-h_window_size, i+h_window_size+1), slice(j-h_window_size, j+h_window_size+1)))
            j += h_window_size
        i += h_window_size
    
    # Separate background and foreground patches
    only_background = np.zeros_like(image)
    only_foreground = np.zeros_like(image)
    foreground_patches = []
    background_patches = []
    foreground_centers = []
    background_centers = []

    for sl in list_slices:
        center = (sl[0].start + h_window_size, sl[1].start + h_window_size)
        if contains_foreground_pixels(label[sl[0], sl[1]]):
            foreground_patches.append(sl)
            foreground_centers.append(center)
        else:
            background_patches.append(sl)
            background_centers.append(center)

    foreground_dm = distance_matrix(foreground_centers, background_centers)

    for fp in foreground_patches:
        only_foreground[fp[0], fp[1]] = image[fp[0], fp[1]]

    for bp in background_patches:
        only_background[bp[0], bp[1]] = image[bp[0], bp[1]]

    # Replace background and foreground patches
    background_patches = np.array(background_patches)
    foreground_patches = np.array(foreground_patches)
    n_closest_patches = 10

    generated_background = only_background.copy()

    for idx, fp in enumerate(foreground_patches):
        # Background patches that are closest to each foreground patch
        closest_ind = np.argsort(foreground_dm[idx])
        closest_background_patches = background_patches[closest_ind][:n_closest_patches]

        # Replace the foreground patch with a random background patch
        random_idx = np.random.randint(0, len(closest_background_patches))
        sl = closest_background_patches[random_idx]
        generated_background[fp[0], fp[1]] = image[sl[0], sl[1]]
    
    return generated_background

def returns_lines_offset_position_size(paths, distance):
  # Algorithm using LineString and OffsetCurve
  
  line_c = LineString(paths)
 
  line_offset_left = line_c.offset_curve(distance=-distance,  join_style=1)
  line_offset_right = line_c.offset_curve(distance=distance, join_style=1)

  largest_size = returns_largest_line(line_offset_left, line_c, line_offset_right)

  return line_offset_left, line_c, line_offset_right, largest_size

def returns_dst_array_np(left_line, center_line, right_line, largest_size):  
  distance = np.linspace(0,1,largest_size)
  dst_array = []
  left_line_points = returns_new_points_from_lines(distance, left_line)
  center_line_points = returns_new_points_from_lines(distance, center_line)
  right_line_points = returns_new_points_from_lines(distance, right_line)

  for l_e in left_line_points:
    dst_array.append(l_e)
  for l_c in center_line_points:
    dst_array.append(l_c)
  for l_d in right_line_points:
    dst_array.append(l_d)
  dst_arr_np = np.array(dst_array)
  return dst_arr_np

def expand_maps_to_trace_size(original_map, largest_value):
  
  rows, cols = original_map.shape
  factor = largest_value/cols
  factor_int = int(factor)  
  remainder = factor - factor_int
  mult = int(remainder*cols)
  
  replicated_image = np.tile(original_map, (1, factor_int))
  rows_rep, cols_rep = replicated_image.shape

  largest_value_int = (cols_rep+mult)

  replicated_image_total = np.zeros((rows, largest_value_int))
  
  replicated_image_total[0:rows_rep,0:cols_rep] = replicated_image
  replicated_image_total[0:rows_rep,cols_rep:largest_value_int] = original_map[0:rows,0:mult]
  
  return replicated_image_total  
 

def delaunay_plot(img, img_out, tri, tri_inv):

    plt.figure(figsize=[100,80])
    ax = plt.subplot(121)
    plt.imshow(img, 'gray')
    x, y = tri.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri.simplices.copy())

    ax = plt.subplot(122)
    plt.imshow(img_out, 'gray')
    x, y = tri_inv.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri_inv.simplices.copy())


def inserting_pot_bottom2(has_pots, img, img_label, background, point, threshold):
    """
    Inserts a pot into the background image at a specified point.
    
    Parameters:
    - has_pots: Binary flag indicating whether pots are present.
    - img: Pot image.
    - img_label: Binary image indicating pot location.
    - background: Background image.
    - point: Coordinates to insert the pot.
    - threshold: Threshold value for image merging.

    Returns:
    - Merged image with the pot inserted.
    - Binary image indicating pot location in the merged image.
    """
    number = 1.8 * 10e100
    merged = np.full(shape=background.shape, fill_value=number)
    img_out_bin_large = np.full(shape=background.shape, fill_value=0)

    img_out_large = np.full(shape=background.shape, fill_value=0)
    rows_img_out_sq, cols_img_out_sq = img.shape

    merged[point[0]:(point[0] + rows_img_out_sq), point[0]:(point[0] + cols_img_out_sq)] = img
    img_out_bin_large[point[0]:(point[0] + rows_img_out_sq), point[0]:(point[0] + cols_img_out_sq)] = img_label
    img_out_large[point[0]:(point[0] + rows_img_out_sq), point[0]:(point[0] + cols_img_out_sq)] = img
    limiar_mask = (merged <= threshold)  # & (has_pots == 0)
    merged[limiar_mask] = background[limiar_mask]
    merged[merged == number] = background[merged == number]
    merged[img_out_bin_large == 1] = img_out_large[img_out_bin_large == 1]

    return merged, img_out_bin_large


def transf_map_dist(img_map, img_map_binary, img_vessel_binary, background):
    """
    Transform the given map based on distance and probabilities.

    Parameters:
    - img_map: Original map image.
    - img_map_binary: Binary representation of the map.
    - img_vessel_binary: Binary representation of vessels.
    - background: Background image.

    Returns:
    - Transformed map image.
    """
    img_copy = img_map.copy()
    rows, cols = img_copy.shape
    img_vessel_binary_sq = img_vessel_binary.squeeze()
    img_dist = ndimage.distance_transform_edt(img_map_binary)
    img_dist[img_vessel_binary_sq] = 0
    img_probs = img_dist / img_dist.max()
    img_probs[img_vessel_binary_sq] = 2
    img_probs[img_map_binary == 0] = 2
    img_rand = np.random.rand(img_map_binary.shape[0], img_map_binary.shape[1])
    inds = np.nonzero(img_rand > img_probs)
    img_copy[inds] = background[0:rows, 0:cols][inds]

    img_copy[img_vessel_binary_sq == 1] = img_map[img_vessel_binary_sq == 1]

    return img_copy


def transform_map_dist2(map, binary_map, binary_vessel, background):
    """
    Transform the given map based on distance and probabilities.

    Parameters:
    - map: Original map image.
    - binary_map: Binary representation of the map.
    - binary_vessel: Binary representation of vessels.
    - background: Background image.

    Returns:
    - Transformed map image.
    """
    # Copy the map image
    img_copy = map.copy()

    # Get the size of the image
    rows, cols = img_copy.shape

    # Convert the vessel image to a one-dimensional array
    vessel_binary_sq = binary_vessel.squeeze()

    # Calculate the distance transformation
    dist_map = ndimage.distance_transform_edt(binary_map)

    # Set the distance to zero where the vessel is present
    dist_map[vessel_binary_sq] = 0

    # Calculate probabilities
    probs = dist_map / dist_map.max()

    # Set specific values for vessel regions and outside the map
    probs[vessel_binary_sq] = 2
    probs[binary_map == 0] = 2

    # Generate a random number matrix
    rand_img = np.random.rand(rows, cols)

    # Find indices where rand_img is greater than probabilities
    inds = np.nonzero(rand_img > probs)

    # Update the copied image with background values where rand_img > probs
    try:
        img_copy[:rows, :cols][inds] = background[:rows, :cols][inds]

        # Update vessel values in the copied image
        img_copy[vessel_binary_sq == 1] = map[vessel_binary_sq == 1]

        return img_copy
    except:
        return None
  

def fill_holes(binary_img_map):
    # Invert the binary image map
    binary_img_inv = 1 - binary_img_map

    # Define a 3x3 structuring element for labeling connected components
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    # Label connected components in the inverted binary image
    labeled_img, num_components = scipy.ndimage.label(binary_img_inv, s)

    # Compute the sum of labels for each connected component
    component_sizes = scipy.ndimage.sum_labels(binary_img_inv, labeled_img, range(1, num_components + 1))

    # Indices of connected components sorted by size
    indices = np.argsort(component_sizes)

    # Fill holes in the binary image map, keeping the two largest connected components
    for idx in indices[:-2]:
        binary_img_map[labeled_img == idx + 1] = 1

    return binary_img_map


def rotate_expanded_map(src_map, dst, max_size):
    # Create a copy of the original map
    original_expanded_map = np.array(src_map)

    # Extract rows and columns of the original expanded map
    rows, cols = original_expanded_map.shape[0], original_expanded_map.shape[1]

    # Create source columns and rows for transformation
    src_cols = np.linspace(0, cols, max_size)
    src_rows = np.linspace(0, rows, 3)
    src_cols, src_rows = np.meshgrid(src_cols, src_rows)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # Perform the transformation using the specified destination points
    img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = transform_v2(src, dst, original_expanded_map)

    return img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin


def create_binary_mask_map(dst, img):
    # Calculate the size of each third in the destination array
    size = len(dst)
    div = size // 3
    div_vector = [div, div * 2, div * 3]

    # Divide the destination array into left and right parts
    right_part = dst[0:div_vector[0]]
    left_part = dst[div_vector[1]:div_vector[2]]

    # Concatenate and create a polygon for binary mask creation
    combined_list = np.concatenate((left_part, right_part[::-1]))
    polygon = np.array(combined_list)[:, ::-1]

    # Create a binary mask for the map
    mask_map = ski.draw.polygon2mask(img.shape, polygon)
    mask_map.astype(int)
    mask_map_sq = mask_map.squeeze()

    return mask_map_sq


def create_binary_mask_vessel(ves_map, new_origin, paths, img):
    # Calculate the minimum and maximum rows for vessel mapping
    min_line = int(np.min(np.rint(ves_map.path2_mapped)))
    max_line = int(np.max(np.rint(ves_map.path1_mapped)))

    # Calculate the maximum vessel diameter
    max_vessel_diameter = (max_line - min_line) / 2

    # Get lines for mask creation and calculate maximum size
    central_line, left_line, right_line, max_size = returns_lines_offset_position_size(paths, max_vessel_diameter)

    # Return the destination array as a NumPy array
    dst_np = returns_dst_array_np(central_line, left_line, right_line, max_size)

    # Adjust the destination array based on the new origin
    dst_np -= new_origin

    # Calculate the size of each third in the destination array
    size = len(dst_np)
    div = size // 3
    div_vector = [div, div * 2, div * 3]

    # Divide the destination array into left and right parts
    right_part = dst_np[0:div_vector[0]]
    left_part = dst_np[div_vector[1]:div_vector[2]]

    # Concatenate and create a polygon for binary mask creation
    combined_list = np.concatenate((left_part, right_part[::-1]))
    polygon = np.array(combined_list)[:, ::-1]

    # Create a binary mask for the vessel
    mask_vessel = ski.draw.polygon2mask(img.shape, polygon)
    mask_vessel.astype(int)
    mask_vessel_sq = mask_vessel.squeeze()

    return mask_vessel_sq

def create_binary_expanded_vessel(binary_map, dst, max_size):
    # Get the dimensions of the binary map
    rows_bin, cols_bin = binary_map.shape[0], binary_map.shape[1]

    # Create source columns and rows for transformation
    src_cols_bin = np.linspace(0, cols_bin, max_size)
    src_rows_bin = np.linspace(0, rows_bin, 3)
    src_cols_bin, src_rows_bin = np.meshgrid(src_cols_bin, src_rows_bin)
    src_bin = np.dstack([src_cols_bin.flat, src_rows_bin.flat])[0]

    # Perform the transformation using the specified destination points
    _, img_out_bin, _, _, _, _, _ = transform_v2(src_bin, dst, binary_map)

    return img_out_bin

def remove_artifacts(img, mask_map):
    # Squeeze the image to 2D
    img_out_sq = img.squeeze()

    # Create an output image with zeros
    img_without_artifacts = np.zeros(img_out_sq.shape, dtype=np.uint8)

    # Iterate through each pixel and copy if the mask is True (white in the boolean image)
    for i in range(img_without_artifacts.shape[0]):
        for j in range(img_without_artifacts.shape[1]):
            if mask_map[i, j] == True:
                img_without_artifacts[i, j] = img_out_sq[i, j]

    return img_without_artifacts

def normalize(img_background, img_map, vessel_mask, threshold):
    # Flatten intensity values of the background image and the map image outside the vessel mask
    ints_background = img_background.flatten()
    ints_map = img_map[vessel_mask == 0]

    # Calculate mean and standard deviation for both background and map intensities
    mean_background = np.mean(ints_background)
    std_background = np.std(ints_background)
    mean_map = np.mean(ints_map)
    std_map = np.std(ints_map)

    # Check if the mean difference is within the specified threshold
    if abs(mean_background - mean_map) > threshold:
        return None
    else:
        # Normalize the map image based on mean and standard deviation of the background
        img_map_norm1 = (img_map - mean_map) / std_map
        img_map_norm = img_map_norm1 * std_background + mean_background
        return img_map_norm

def histogram_matching(img_map, img_vessel_label, img_background):
    # Squeeze the vessel label image to 2D
    img_vessel_label_sq = img_vessel_label.squeeze()

    # Create a copy of the map image
    map_copy = img_map.copy()

    # Identify vessel positions in the label and set them to 0 in the copy
    vessel_positions = (img_vessel_label_sq == 1)
    map_copy[vessel_positions] = 0

    # Perform histogram matching between the copy and the background image
    matched_histogram = ski.exposure.match_histograms(map_copy, img_background)

    # Restore the vessel positions in the matched histogram
    matched_histogram[vessel_positions] = img_map[vessel_positions]

    return matched_histogram
 
 
def insert_map(background, img_vessel_bin, img_map, img_map_bin, threshold, has_maps):
    # Create a copy of the background image
    merged_map = background.copy()
    img_map_copy = img_map.copy()
    
    # Get dimensions of the map image
    rows, cols = img_map.shape

    # Create a binary mask based on threshold, vessel mask, and map binary
    threshold_mask = (img_map <= threshold) & (img_map_bin == 1) & (img_vessel_bin == 0)
    
    # Update the map copy with background values where the mask is True
    img_map_copy[0:rows, 0:cols][threshold_mask] = background[0:rows, 0:cols][threshold_mask]

    # Get pixel coordinates where the map binary is True and has_maps is False
    pix_map = np.nonzero(img_map_bin & (has_maps[0:rows, 0:cols] == 0))
    
    # Get pixel coordinates where the vessel binary is True
    pix_vessel = np.nonzero(img_vessel_bin)
    
    # Update the has_maps array to indicate the presence of maps
    has_maps[pix_vessel] += 1
    
    # Update the merged map with values from the map copy at map binary True positions
    merged_map[pix_map] = img_map_copy[pix_map]

    return merged_map

def insert_binary_map(background, img_vessel_bin, has_maps):
    # Create a copy of the background image
    merged_map = background.copy()
    img_vessel_copy = img_vessel_bin.copy()

    # Get pixel coordinates where the vessel binary is True
    pix_vessel = np.nonzero(img_vessel_bin)

    # Update the has_maps array to indicate the presence of maps
    has_maps[pix_vessel] += 1

    # Update the merged map with values from the vessel copy at vessel binary True positions
    merged_map[pix_vessel] = img_vessel_copy[pix_vessel]

    return merged_map

def insert_vessels(medial_path_array, distance, pickles_array, pickle_dir, back_artifact, threshold, path_pickle=None):
    # Choose a random pickle file if path_pickle is not provided
    if path_pickle is not None:
        path = path_pickle
    else:
        n_random = np.random.randint(0, len(pickles_array))
        path = (pickle_dir + f'/{pickles_array[n_random]}')

    # Load the pickle file
    pickle_file = pickle.load(open(path, 'rb'))
    vessel_map = pickle_file['vessel_model'].vessel_map
    original_map = vessel_map.mapped_values

    # Get the original binary map without lateral artifacts
    binary_map_original = vessel_map.mapped_mask_values
    binary_map_without_lateral_artifacts = returns_binary_image_without_artifacts(vessel_map, binary_map_original)

    # Return None if binary map without lateral artifacts is None
    if binary_map_without_lateral_artifacts is None:
        return None

    # Fill holes in the binary map without lateral artifacts
    binary_map_without_artifacts = fill_holes(binary_map_without_lateral_artifacts)

    # Normalize the original map based on background artifact, binary map without artifacts, and a threshold
    normalized_original_map = normalize(back_artifact, original_map, binary_map_without_artifacts, threshold)

    # Return None if normalized map is None
    if normalized_original_map is None:
        return None

    # Get dimensions of the original map
    rows, cols = original_map.shape

    # Calculate the distance to be used for the map expansion
    distance = (rows / 2)

    # Find the threshold based on the most frequent pixel value in the normalized map
    threshold1 = find_most_frequent_pixel(normalized_original_map)

    # Set the maximum value for expansion
    max_value = int(distance)

    # Expand the original map to the specified size
    expanded_original_map = expand_maps_to_trace_size(normalized_original_map, max_value)

    # Expand the binary vessel map to the specified size
    expanded_vessel_bin = expand_maps_to_trace_size(binary_map_without_artifacts, max_value)

    # Get left, central, and right lines for the expansion from the medial path
    left_offset_line, central_line, right_offset_line, max_size = returns_lines_offset_position_size(
        medial_path_array, distance)

    # Create destination array for transformation from left, central, and right lines
    dst_array_np = returns_dst_array_np(left_offset_line, central_line, right_offset_line, max_size)

    # Execute the algorithm that transforms the expanded map
    img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = rotate_expanded_map(
        expanded_original_map, dst_array_np, max_size)

    # Create a binary mask for the map
    mask_map = create_binary_mask_map(new_dst, img_out)

    # Create a binary mask for the vessel
    mask_vessel = create_binary_mask_vessel(vessel_map, new_origin, medial_path_array, img_out)

    # Remove artifacts from the map
    map_without_artifacts = remove_artifacts(img_out, mask_map)

    # Get dimensions of the map without artifacts
    rows_art, cols_art = map_without_artifacts.shape

    # Get dimensions of the background artifact
    rows_back, cols_back = back_artifact.shape

    # Return None if the dimensions of the map without artifacts are greater than or equal to the background artifact
    if rows_art >= rows_back or cols_art >= cols_back:
        return None

    # Create an expanded and rotated binary vessel map
    img_out_bin = create_binary_expanded_vessel(expanded_vessel_bin, dst_array_np, max_size)

    # Remove artifacts from the binary vessel map
    vessel_without_artifacts = remove_artifacts(img_out_bin, mask_vessel)

    # Transform the map without artifacts
    map_without_artifacts_transf = transform_map_dist2(map_without_artifacts, mask_map, vessel_without_artifacts,
                                                        back_artifact)

    # Return the result if the transformed map without artifacts is not None
    if map_without_artifacts_transf is not None:
        return vessel_without_artifacts, map_without_artifacts_transf, mask_map, threshold1
    else:
        return vessel_without_artifacts, map_without_artifacts, mask_map, threshold1

# results = backgen.insert_vessels2(vector_medial_path[0], vector_medial_path[1], 
def insert_vessels2(medial_path_array, distance, vessel_map, normalized_original_map, binary_map_without_artifacts, back_artif):   
    
    # Get dimensions of the original map
    rows, cols = normalized_original_map.shape

    # Calculate the distance to be used for the map expansion
    distance = (rows / 2)

    # Find the threshold based on the most frequent pixel value in the normalized map
    threshold1 = find_most_frequent_pixel(normalized_original_map)

    # Set the maximum value for expansion
    max_value = int(distance)

    # Expand the original map to the specified size
    expanded_original_map = expand_maps_to_trace_size(normalized_original_map, max_value)

    # Expand the binary vessel map to the specified size
    expanded_vessel_bin = expand_maps_to_trace_size(binary_map_without_artifacts, max_value)

    # Get left, central, and right lines for the expansion from the medial path
    left_offset_line, central_line, right_offset_line, max_size = returns_lines_offset_position_size(
        medial_path_array, distance)

    # Create destination array for transformation from left, central, and right lines
    dst_array_np = returns_dst_array_np(left_offset_line, central_line, right_offset_line, max_size)

    # Execute the algorithm that transforms the expanded map
    img_proper, img_out, new_src, new_dst, tform_out, translation, new_origin = rotate_expanded_map(
        expanded_original_map, dst_array_np, max_size)

    # Create a binary mask for the map
    mask_map = create_binary_mask_map(new_dst, img_out)

    # Create a binary mask for the vessel
    mask_vessel = create_binary_mask_vessel(vessel_map, new_origin, medial_path_array, img_out)

    # Remove artifacts from the map
    map_without_artifacts = remove_artifacts(img_out, mask_map)

    # Get dimensions of the map without artifacts
    rows_art, cols_art = map_without_artifacts.shape

    # Get dimensions of the artificial background
    rows_back, cols_back = back_artif.shape

    #import pdb; pdb.set_trace()
    # Return None if the dimensions of the map without artifacts are greater than or equal to the background artifact
    #if rows_art >= rows_back or cols_art >= cols_back:
        #return None

    # Create an expanded and rotated binary vessel map
    img_out_bin = create_binary_expanded_vessel(expanded_vessel_bin, dst_array_np, max_size)

    # Remove artifacts from the binary vessel map
    vessel_without_artifacts = remove_artifacts(img_out_bin, mask_vessel)

    # Transform the map without artifacts
    map_without_artifacts_transf = transform_map_dist2(map_without_artifacts, mask_map, vessel_without_artifacts,
                                                        back_artif)

    # Return the result if the transformed map without artifacts is not None
    if map_without_artifacts_transf is not None:
        return vessel_without_artifacts, map_without_artifacts_transf, mask_map, threshold1
    else:
        return vessel_without_artifacts, map_without_artifacts, mask_map, threshold1