import sys
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")

import numpy as np
from matplotlib.patches import Arrow, ArrowStyle, FancyArrow
from matplotlib.collections import PatchCollection
from scipy.ndimage import map_coordinates
from Slice_mapper import slice_mapper_util as smutil
from shapely import geometry, ops as shops, affinity
from IPython.display import display
from skimage import draw

# Creation of classes to facilitate encapsulation of variables, attributes, and functions.
class SliceMapper:
    """Class that generates the vessel model and map. Calls functions that create the VesselModel and VesselMap.

    Parameters:
    -----------
    img: ndarray, float
        Original image.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    reach: float
        Variable that delimits the size of the vessel map. Sets the upper and lower bounds that the map will cover.
    
    Methods:
    -----------
    add_model:
        Creates the vessel model and map.
    """
    def __init__(self, img, delta_eval, smoothing, reach):
        """Initialize the class with input parameters.

        Attributes:
        -----------
        img: ndarray, float
            Original image.
        delta_eval: float
            Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
        smoothing: float
            Smoothing criterion.
        reach: float
            Variable that delimits the size of the vessel map. Sets the upper and lower bounds that the map will cover.
        models: list
            List to store vessel models.
        debug: list
            List for storing debug information.
        """
        self.img = img
        self.delta_eval = delta_eval
        self.smoothing = smoothing
        self.reach = reach
        self.models = []
        self.debug = []

    def add_model(self, path1, path2, generate_map=True):
        """Add a vessel model to the list.

        Parameters:
        -----------
        path1: 
            First path parameter.
        path2: 
            Second path parameter.
        generate_map: bool, optional
            Flag to generate the vessel map. Default is True.
        """
        vessel_model = create_vessel_model(self.img, path1, path2, self.delta_eval, self.smoothing)

        if generate_map:
            vessel_map = create_map(self.img, vessel_model, self.reach,
                                    self.delta_eval, self.smoothing)
            vessel_model.set_map(vessel_map)

        self.models.append(vessel_model)



class VesselModel:
    """Class that stores information related to the vessel model.

    Parameters:
    -----------
    path1: ndarray, float
        Vector of path 1.
    path1_info: tuple
        Information about path 1 is stored in NumPy arrays and stored in path1_info.
    path2: ndarray, float
        Vector of path 2.
    path2_info: tuple
        Information about path 2 is stored in NumPy arrays and stored in path2_info.
    medial_path: ndarray, float
        Medial path.
    medial_info: ndarray, float
        Information about the medial line is stored in NumPy arrays and stored in medial_info.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    vessel_map: object VesselMap
        Instance of the VesselMap object.
    img_file: ndarray, float
        Image file.
    Return:
    -----------
        Absorbs the information passed in the constructor and stores it in the VesselModel object.
    """

    def __init__(self, path1, path1_info, path2, path2_info, medial_path, medial_info,
                 delta_eval, vessel_map=None, img_file=None):
        self.path1 = {
            'original': path1,
            'interpolated': path1_info[0],
            'tangents': path1_info[1],
            'normals': path1_info[2],
        }

        self.path2 = {
            'original': path2,
            'interpolated': path2_info[0],
            'tangents': path2_info[1],
            'normals': path2_info[2],
        }

        self.medial_path = {
            'original': medial_path,
            'interpolated': medial_info[0],
            'tangents': medial_info[1],
            'normals': medial_info[2],
        }
        self.delta_eval = delta_eval
        self.vessel_map = vessel_map
        self.img_file = img_file

    def set_map(self, vessel_map):
        self.vessel_map = vessel_map


class VesselMap:
    """Class that stores information related to the vessel map.

    Parameters:
    -----------
    mapped_values: ndarray, float
        Mapped values.
    medial_coord: ndarray, float
        Medial coordinates.
    cross_coord: ndarray, float
        Transversal coordinates.
    cross_versors: list, float
        List containing transversal versors.
    mapped_mask_values: ndarray, float
        Mapped values in binary.
    path1_mapped: ndarray, float
        Mapped path 1.
    path2_mapped: ndarray, float
        Mapped path 2.
    Return:
    -----------
        Absorbs the information passed in the constructor and stores it in the VesselMap object.
    """

    def __init__(self, mapped_values, medial_coord, cross_coord, cross_versors, mapped_mask_values,
                 path1_mapped, path2_mapped):
        self.mapped_values = mapped_values
        self.medial_coord = medial_coord
        self.cross_coord = cross_coord
        self.cross_versors = cross_versors
        self.mapped_mask_values = mapped_mask_values
        self.path1_mapped = path1_mapped
        self.path2_mapped = path2_mapped



def interpolate_envelop(path1, path2, delta_eval=2., smoothing=0.01):
    """Wraps items, path1 and path2, their smoothed interpolations, their tangents, and their normals.

    Parameters:
    -----------
    path1: ndarray, float
        Path 1 vector.
    path2: ndarray, float
        Path 2 vector.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    Return:
    -----------  
    path1: ndarray, float
        Path 1.
    path1_interp: ndarray, float
        Interpolated and smoothed Path 1.
    tangents1: ndarray, float
        Tangent vector of Path 1.
    normals1: ndarray, float
        Normal vector of Path 1.
    path2: ndarray, float
        Path 2.
    path2_interp: ndarray, float
        Interpolated and smoothed Path 2.
    tangents2: ndarray, float
        Tangent vector of Path 2.
    normals2: ndarray, float
       Normal vector of Path 2.
    """

    # Paths are interpolated, and new tangents are created from the interpolation of the paths.
    path1_interp, tangents1 = smutil.two_stage_interpolate(path1, delta_eval=delta_eval, smoothing=smoothing)
    path2_interp, tangents2 = smutil.two_stage_interpolate(path2, delta_eval=delta_eval, smoothing=smoothing)

    # Normal vectors are created from the new tangents.
    normals1 = smutil.get_normals(tangents1)
    normals2 = smutil.get_normals(tangents2)

    min_size = min([len(path1_interp), len(path2_interp)])

    # Make the normals point in opposite directions.
    congruence = np.sum(np.sum(normals1[:min_size] * normals2[:min_size], axis=1))
    if congruence > 0:
        normals2 *= -1

    # Make the normals point towards the interior.
    vsl1l2 = path2_interp[:min_size] - path1_interp[:min_size]
    congruence = np.sum(np.sum(vsl1l2 * normals1[:min_size], axis=1))
    if congruence < 0:
        normals1 *= -1
        normals2 *= -1

    if np.cross(tangents1[1], normals1[1]) < 0:
        # Make path1 run to the left of path2.
        path1, path2 = path2, path1
        path1_interp, path2_interp = path2_interp, path1_interp
        tangents1, tangents2 = tangents2, tangents1
        normals1, normals2 = normals2, normals1

    return path1, (path1_interp, tangents1, normals1), path2, (path2_interp, tangents2, normals2)


def extract_medial_path(path1_interp, path2_interp, delta_eval=2., smoothing=0.01, return_voronoi=False):
    """Extracts the medial path from a tubular structure.

    Parameters:
    -----------
    path1_interp: ndarray, float
        Interpolated Path 1.
    path2_interp: ndarray, float
        Interpolated Path 2.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    return_voronoi: boolean
        When True, returns information about the created Voronoi object.

    Return:
    -----------
    medial_path: ndarray, float
        Medial path.
    medial_path_info: ndarray, float
        Contains the medial path, its tangents, and its normals.
    vor: Voronoi object
        Returns information about the Voronoi object.
    """
    vor, idx_medial_vertices, point_relation = smutil.medial_voronoi_ridges(path1_interp, path2_interp)
    idx_medial_vertices_ordered = smutil.order_ridge_vertices(idx_medial_vertices)
    medial_path = []
    for idx_vertex in idx_medial_vertices_ordered:
        medial_path.append(vor.vertices[idx_vertex])
    medial_path = np.array(medial_path)
    medial_path = smutil.invert_if_opposite(path1_interp, medial_path)

    # Ensure that the medial path goes to the end of the tube.
    # Take the average of the interpolated paths.
    first_point = (path1_interp[0] + path2_interp[0]) / 2
    last_point = (path1_interp[-1] + path2_interp[-1]) / 2
    medial_path = np.array([first_point.tolist()] + medial_path.tolist() + [last_point.tolist()])

    # Interpolate the medial path for smoothing.
    medial_path_info = interpolate_medial_path(medial_path, delta_eval=delta_eval, smoothing=smoothing)

    if return_voronoi:
        return medial_path, medial_path_info, vor
    else:
        return medial_path, medial_path_info


def create_vessel_model(img, path1, path2, delta_eval, smoothing):
    """This function creates the vessel model.

    Parameters:
    -----------
    img: ndarray, float
        Image that serves as the basis for creating the vessel model.
    path1: ndarray, float
        Path 1.
    path2: ndarray, float
        Path 2.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    Return:
    -----------
    vm: object VesselModel
        Returns the vessel model with an instantiated object of the VesselModel class.
    """

    # Call the inversion function. If the path is inverted, path2 is also inverted.
    path2 = smutil.invert_if_opposite(path1, path2)

    # Variables absorb the result of enveloping path1, path2, passing a delta_eval 
    # that increases resolution, and applying a degree of smoothing.
    path1, path1_info, path2, path2_info = interpolate_envelop(path1, path2, delta_eval, smoothing)

    # Information contained in paths 1 and 2 is inserted into variables.
    path1_interp, tangents1, normals1 = path1_info
    path2_interp, tangents2, normals2 = path2_info

    # The medial path, along with its information, is created.
    medial_path, medial_path_info = extract_medial_path(path1_interp, path2_interp, delta_eval=delta_eval,
                                                        smoothing=smoothing)

    # The vessel model is created and returned from the function.
    # Instantiation of the VesselModel class containing path 1, information about path 1, path 2, information about path 2,
    # the medial path, information about the medial path, and delta_eval.
    vm = VesselModel(path1, path1_info, path2, path2_info, medial_path, medial_path_info, delta_eval)

    return vm


def create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=False):
    """Creates an image containing intensities of cross-sectional slices along the provided medial path.

    Parameters:
    -----------
    img: ndarray, float
        Image that serves as the basis for creating the map.
    vessel_model: object VesselModel
        Object of type VesselModel.
    reach: float
        Variable that defines how much upper and lower limit the image will have. It has a direct impact on the number of lines in the created map.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    return_cross_paths: boolean
        By default, it is False. If True, returns valid transverse paths.
    Return:
    -----------
    vessel_map: object VesselMap
        Returns the vessel map as an instantiated object of the VesselMap class.
    cross_paths_valid: ndarray
        Returns valid transverse paths.
    """

    # Paths absorb the values of the vessel model at the 'interpolated' index.
    path1_interp = vessel_model.path1['interpolated']
    path2_interp = vessel_model.path2['interpolated']

    # The interpolated medial path and medial normals are created from the vessel model.
    medial_path_interp, medial_normals = vessel_model.medial_path['interpolated'], vessel_model.medial_path['normals']

    # Transverse coordinates are created based on reach (height) and delta_eval, concatenating the values into an array.
    cross_coord = np.concatenate((np.arange(-reach, 0 + 0.5 * delta_eval, delta_eval),
                                  np.arange(delta_eval, reach + 0.5 * delta_eval, delta_eval)))

    # Transverse paths and transverse versors are created using the function to create transverse paths.
    cross_paths, cross_versors = create_cross_paths(cross_coord, medial_path_interp, medial_normals, path1_interp,
                                                    path2_interp, reach)

    # The medial coordinate is created by smoothing the arc length of the interpolated medial path.
    medial_coord = smutil.arc_length(medial_path_interp)

    cross_paths_valid = []

    # Function that takes the entire crossed path, checks if it is empty, and adds the valid values to a vector
    # of valid crossed paths.
    for idx, cross_path in enumerate(cross_paths[1:-1], start=1):
        if cross_path is not None:
            cross_paths_valid.append(cross_path)
    cross_paths_valid = np.array(cross_paths_valid)

    # Variable that absorbs the flat transverse paths from the points in the transverse path.
    cross_paths_flat = np.array([point for cross_path in cross_paths_valid for point in cross_path])

    # Mapping values are calculated using the map_coordinates method from scipy.ndimage, passing some parameters
    # and the transposed flat transverse paths.
    mapped_values = map_coordinates(img.astype(float), cross_paths_flat.T[::-1], output=float, mode='mirror')

    # The mapped paths are reshaped and transposed.
    mapped_values = mapped_values.reshape(-1, len(cross_coord)).T

    # Generation of a mask for the image and for the mapped values.
    # It will replace the original image with a binary image containing only the vessel.
    mask_img = generate_mask(path1_interp, path2_interp, img.shape)

    # The binary mapped values are created.
    mapped_mask_values = map_coordinates(mask_img, cross_paths_flat.T[::-1], output=np.uint8,
                                         order=0, mode='mirror')
    mapped_mask_values = mapped_mask_values.reshape(-1, len(cross_coord)).T

    # Get the precise positions for path1 and path2 interpolated in the map.
    path1_mapped, path2_mapped = find_vessel_bounds_in_map(path1_interp,
                                                           path2_interp, cross_paths_valid, delta_eval, smoothing)

    # Instantiation of the VesselMap class object, storing the mapped values, medial coordinates, transverse coordinates,
    # transverse versors, binary mapped values, path 1 and 2 mapped.
    vessel_map = VesselMap(mapped_values, medial_coord, cross_coord, cross_versors, mapped_mask_values, path1_mapped,
                           path2_mapped)
    if return_cross_paths:
        return vessel_map, cross_paths_valid
    else:
        return vessel_map


def find_vessel_bounds_in_map(path1_interp, path2_interp, cross_paths, delta_eval, smoothing):
    """Finds the vessel bounds on the map.

    Parameters:
    -----------
    path1_interp: ndarray, float
        Interpolated path 1.
    path2_interp: ndarray, float
        Interpolated path 2.
    cross_paths: ndarray
        Array containing transverse paths.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    Return:
    -----------
    path1_mapped: list, float
        List containing the mapping of path 1.
    path2_mapped: list, float
        List containing the mapping of path 2.
    """

    # LineString: The constructed LineString object represents one or more connected linear splines between points.
    # Repeated points in the ordered sequence are allowed but may incur performance penalties and should be avoided.
    # A LineString can intersect, i.e., be complex and not simple.
    sh_path1_interp = geometry.LineString(path1_interp)
    sh_path2_interp = geometry.LineString(path2_interp)
    path1_mapped = []
    path2_mapped = []

    # Iterate over transverse paths.
    for cross_path in cross_paths:

        # Apply LineString to the transverse path.
        sh_cross_path = geometry.LineString(cross_path)

        # Path limit is obtained through the intersections of crossed paths.
        path_lim = find_envelop_cross_path_intersection(sh_cross_path, sh_path1_interp)
        if path_lim is None:
            path1_mapped.append(np.nan)
        else:
            # sh_path1_cross_coord receives the return of the distance along this geometric object to the nearest point of the other object.
            sh_path1_cross_coord = sh_cross_path.project(path_lim)
            path1_mapped.append(np.array(sh_path1_cross_coord))
        path_lim = find_envelop_cross_path_intersection(sh_cross_path, sh_path2_interp)

        # The same procedure is done for path 2.
        if path_lim is None:
            path2_mapped.append(np.nan)
        else:
            sh_path2_cross_coord = sh_cross_path.project(path_lim)
            path2_mapped.append(np.array(sh_path2_cross_coord))

    # The smaller the delta_eval value, the greater the number of mapped values.
    path1_mapped = np.array(path1_mapped) / delta_eval
    path2_mapped = np.array(path2_mapped) / delta_eval

    # Return lists containing the mapped values of path 1 and 2.
    return path1_mapped, path2_mapped


def find_envelop_cross_path_intersection(sh_cross_path, sh_path_interp, max_dist_factor=2.):
    """Finds intersections of transverse paths of the envelope.

    Parameters:
    -----------
    sh_cross_path: object, LineString
        Object built from the shapely.geometry.linestring.LineString class.
    sh_path_interp: object, LineString
        Object built from the shapely.geometry.linestring.LineString class.
    max_dist_factor: float
        Parameter defining the factor of the greatest distance.
    Return:
    -----------
    path_lim: object, Point
        Object built from the shapely.geometry.point.Point class.
    """

    # Get the integer index of the middle of the sh_cross_path.coords length.
    idx_middle_cross_point = len(sh_cross_path.coords) // 2

    # The path limit is obtained through the intersections of sh_cross_path.
    path_lim = sh_path_interp.intersection(sh_cross_path)
    if path_lim.is_empty:
        # At the endpoints, paths may not intersect.
        path_lim = None
    else:
        sh_middle_cross_point = geometry.Point(sh_cross_path.coords[idx_middle_cross_point])
        if path_lim.geom_type == 'MultiPoint':
            # Paths intersect at more than one point, it is necessary to find the point closest to the middle.
            distances = []
            for point in path_lim.geoms:
                distances.append(sh_middle_cross_point.distance(point))
            path_lim = path_lim.geoms[np.argmin(distances)]

        min_distance = sh_middle_cross_point.distance(sh_path_interp)
        distance_path_lim = sh_middle_cross_point.distance(path_lim)
        if distance_path_lim > max_dist_factor * min_distance:
            path_lim = None

    # Return the path limit.
    return path_lim


def map_slices(img, path1, path2, delta_eval, smoothing, reach):
    """Creating vessel models and maps.

    Parameters:
    -----------
    path1: ndarray, float
        Path 1 vector.
    path2: ndarray, float
        Path 2 vector.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    reach: float
        Variable that delimits the size of the vessel map. Sets the upper and lower bounds that the
        map will cover.
    Return:
    -----------
    vessel_model: object VesselModel
        Returns the vessel model with an instantiated object of the VesselModel class.
    cross_paths: ndarray, float
        Transverse paths.
    """

    # Creation of the vessel model.
    vessel_model = create_vessel_model(img, path1, path2, delta_eval, smoothing)

    # Creation of the vessel map and transverse paths.
    vessel_map, cross_paths = create_map(img, vessel_model, reach, delta_eval, smoothing, return_cross_paths=True)
    vessel_model.set_map(vessel_map)

    # Returning the vessel model and transverse paths.
    return vessel_model, cross_paths



def interpolate_medial_path(path, delta_eval=2., smoothing=0.01):
    """Interpolates the medial path.

    Parameters:
    -----------
    path: ndarray, float
        Path vector.
    delta_eval: float
        Parameter that increases resolution and creates intermediate points between coordinates (interpolates).
    smoothing: float
        Smoothing criterion.
    Return:
    -----------
    path_interp: ndarray, float
        Interpolated path.
    tangents: ndarray, float
        Vector containing tangents.
    normals: ndarray, float
        Vector containing normals.
    """

    # Interpolated path and tangents are calculated from two stages of interpolation.
    # The first stage is linear, and the second is cubic.
    path_interp, tangents = smutil.two_stage_interpolate(path, delta_eval=delta_eval, smoothing=smoothing)

    # Normals are obtained from tangents.
    normals = smutil.get_normals(tangents)
    if np.cross(tangents[0], normals[0]) > 0:
        # Making the normals point to the "left" of the medial_path.
        normals *= -1

    # Returning the interpolated path, tangents, and normals.
    return path_interp, tangents, normals


def show_interpolated(path_interp, tangents, normals, ax, scale=2., color='blue'):
    """Shows the interpolated path along with tangents and normals. The scale parameter defines the length
    of the arrows.

    Parameters:
    -----------
    path_interp: ndarray, float
        Interpolated path.
    tangents: ndarray, float
        Vector containing tangents.
    normals: ndarray, float
        Vector containing normals.
    ax: object, AxesSubplot
        Object of type AxesSubplot from the matplotlib library.
    scale: float
        Parameter that sets the scale to be followed.
    color: str
        String storing the color for displaying the interpolated path.
    Return:
    -----------
        The interpolated paths will be plotted, and tangent and normal columns will be added to the axes.
    """

    # Definition of the length of the heads of tangents and normals.
    tangent_heads = path_interp + scale * tangents
    normals_heads = path_interp + scale * normals

    # Arrow style.
    arrow_style = ArrowStyle("->", head_length=10, head_width=3)

    # Tangent arrows vector.
    tangent_arrows = []
    for idx in range(len(path_interp)):
        # fa receives the FancyArrow method, which is part of the matplotlib library, which, when passing a polygon
        # as a parameter, creates an arrow.
        fa = FancyArrow(path_interp[idx, 0], path_interp[idx, 1], scale * tangents[idx, 0], scale * tangents[idx, 1],
                        width=0.01, head_width=0.1, head_length=0.2, color='orange')
        # The tangent arrows vector is incremented.
        tangent_arrows.append(fa)
    tangents_col = PatchCollection(tangent_arrows, match_original=True, label='Tangent')

    # Normal arrows vector.
    normal_arrows = []
    for idx in range(len(path_interp)):
        # fa receives the FancyArrow method, which is part of the matplotlib library, which, when passing a polygon
        # as a parameter, creates an arrow.
        fa = FancyArrow(path_interp[idx, 0], path_interp[idx, 1], scale * normals[idx, 0], scale * normals[idx, 1],
                        width=0.01, head_width=0.1, head_length=0.2, color='orange')
        normal_arrows.append(fa)
    # The matplotlib PatchCollection takes the normal arrows and adds them to normals_col. PatchCollection stores a set of patches, which in
    # this case are the set of normal arrows.
    normals_col = PatchCollection(normal_arrows, match_original=True, label='Normal')

    # Plotting the interpolated paths.
    ax.plot(path_interp[:, 0], path_interp[:, 1], '-', c=color, label='Interpolated')

    # Adding tangent and normal columns to the axes.
    ax.add_collection(tangents_col)
    ax.add_collection(normals_col)

x = 0
def plot_model(img, vessel_model, cross_paths, ax):

    from matplotlib import pyplot as plt
    """Plotting the image along with the vessel model, with filled lines along the vessel, upper
    and lower, in green, and display of the medial line in red.

    Parameters:
    -----------
    img: ndarray, float
        Image of the area containing the vessel.
    vessel_model: object VesselModel
       Returns the vessel model with an instantiated object of the VesselModel class.
    cross_paths: ndarray, float
       Transverse paths.
    ax: object, AxesSubplot
       Object of type AxesSubplot from the matplotlib library.
    Return:
    -----------
       Interpolated paths 1, 2 (green color) and medial line (red color) are displayed, along with tangents and normals for each of these items.
    """
    # Variables that absorb paths 1 and 2 from the vessel model.
    p1_data = vessel_model.path1
    p2_data = vessel_model.path2

    # Absorbing the medial path from the vessel model.
    medial_data = vessel_model.medial_path

    # set_aspect with the equal parameter ensures that the x and y axes have the same scale.
    ax.set_aspect('equal')
    ax.imshow(img, 'gray')

    # Calling the function that shows the interpolated data, tangents, and normals.
    show_interpolated(p1_data['interpolated'], p1_data['tangents'], p1_data['normals'], ax,
                      scale=0.6, color='green')
    show_interpolated(p2_data['interpolated'], p2_data['tangents'], p2_data['normals'], ax,
                      scale=0.6, color='green')
    show_interpolated(medial_data['interpolated'], medial_data['tangents'], medial_data['normals'], ax,
                      scale=0.6, color='red')
    
    #plt.savefig(f'img{x}.svg',format='svg')



def generate_mask(path1, path2, img_shape):
    """Function that transforms values into binary.

    Parameters:
    -----------
    path1: ndarray, float
        Vector of path 1.
    path2: ndarray, float
        Vector of path 2.
    img_shape: tuple, int
        Informs the number of rows and columns that the vessel model image will contain.
    Return:
    -----------
    mask_img: ndarray, containing True and False values
        Returns the mask for the input polygon, which in this case is path1, path2, and img_shape.
    """

    # concatenate ==> joins a sequence of vectors along the rows
    envelop = np.concatenate((path1, path2[::-1]), axis=0)

    # round ==> rounds a matrix, converts it to an integer
    envelop = np.round(envelop).astype(int)[:, ::-1]

    # transforms the image into binary, passing the shape of the image and the envelop (polygon) created
    mask_img = draw.polygon2mask(img_shape, envelop)
    return mask_img


def create_cross_paths(cross_coord, medial_path, medial_normals, path1, path2, reach, normal_weight=2,
                       path_res_factor=3, angle_limit=45, angle_res=2):
    """Functions related to the creation of transverse paths.

    Parameters:
    -----------
    cross_coord: ndarray, float
        Vector containing transverse coordinates.
    medial_path: ndarray, float
        Medial path.
    medial_normals: ndarray, float
        Medial path normals.
    path1: ndarray, float
        Vector of path 1.
    path2: ndarray, float
        Vector of path 2.
    reach: float
        Variable that delimits the size of the vessel map. Sets the upper and lower bounds that the
        map will cover.
    normal_weight: int
        Height of the normals.
    path_res_factor: int
        Value that determines how much the resolution of the path will be increased. The higher this value, the more points will be
        created.
    angle_limit: int
        Value that determines the angle limit.
    angle_res: int
        Determines the variation that the angle will have.
    Return:
    -----------
    cross_paths: list, float
        List containing values of transverse paths.
    cross_versors: list, float
        List containing values of transverse versors.
    """

    # creation of transverse versors
    # creates vectors most aligned with the normals of the envelope lines and the medial line
    cross_versors = create_cross_versors(medial_path, medial_normals, path1, path2, reach, normal_weight,
                                         path_res_factor, angle_limit, angle_res)

    # transpose of transverse coordinates
    cross_coord = cross_coord[None].T

    cross_paths = []
    # function that takes the indices and points of the medial path to create transverse paths
    for idxm, pointm in enumerate(medial_path):

        # takes the index in the crossed versors
        cross_versor = cross_versors[idxm]

        # if the transverse versor is empty, the transverse paths receive None
        if cross_versor is None:
            cross_paths.append(None)
        else:
            # the transverse path receives the point + the transverse coordinate multiplied by the transverse versor
            # absorbs the values and inserts the points in a transverse line
            cross_path = pointm + cross_coord * cross_versor
            # the transverse paths add the transverse path in list format
            cross_paths.append(cross_path.tolist())

            # return of the transverse paths and the transverse versors
    return cross_paths, cross_versors


def create_cross_versors(medial_path, medial_normals, path1, path2, reach, normal_weight=2,
                         path_res_factor=3, angle_limit=45, angle_res=2):
    """Function that creates transverse versors.

    Parameters:
    -----------
    medial_path: ndarray, float
        Medial path.
    medial_normals: ndarray, float
        Medial path normals.
    path1: ndarray, float
        Vector of path 1.
    path2: ndarray, float
        Vector of path 2.
    reach: float
        Variable that delimits the size of the vessel map. Sets the upper and lower bounds that the
        map will cover.
    normal_weight: int
        Height of the normals.
    path_res_factor: int
        Value that determines how much the resolution of the path will be increased. The higher this value, the more points will be
        created.
    angle_limit: int
        Value that determines the angle limit.
    angle_res: int
        Determines the variation that the angle will have.
    Return:
    -----------
    cross_versors: list, float
        List containing values of transverse versors.
    """

    # definition of angles -
    # concatenate ==> joins a sequence of vectors arranged. 
    # The vectors have an angle limit of 45, and the others have size 2
    angles = np.concatenate((np.arange(-angle_limit, 0 + 0.5 * angle_res, angle_res),
                             np.arange(0, angle_limit + 0.5 * angle_res, angle_res)))

    # calls the function to find the best angles
    idx_best_angles = find_best_angles(medial_path, medial_normals, path1, path2, angles, reach,
                                       normal_weight, path_res_factor)

    cross_versors = []

    # function that takes the indices and points of the medial path to create transverse versors
    for idxm, pointm in enumerate(medial_path):

        # takes the index of the best angles
        idx_best_angle = idx_best_angles[idxm]

        # verification if the index of the best angle is None
        if idx_best_angle is None:
            cross_versors.append(None)
        else:
            # creation of the normal from
            normalm = medial_normals[idxm]
            sh_normalm = geometry.Point(normalm)
            # does the rotation by finding the best angles
            sh_normalm_rotated = affinity.rotate(sh_normalm, angles[idx_best_angle], origin=(0, 0))
            normalm_rotated = np.array(sh_normalm_rotated.coords)[0]
            cross_versors.append(normalm_rotated)
    return cross_versors


def find_best_angles(medial_path, medial_normals, path1, path2, angles, reach, normal_weight=2,
                     path_res_factor=3):
    """Function that finds the best angles. Rotates if necessary.

    Parameters:
    -----------
    medial_path: ndarray, float
        Medial path.
    medial_normals: ndarray, float
        Medial path normals.
    path1: ndarray, float
        Vector of path 1.
    path2: ndarray, float
        Vector of path 2.
    angles: ndarray, float
        Vector that absorbs the upper and lower angle limit values.
    reach: float
        Variable that delimits the size of the vessel map. Sets the upper and lower bounds that the
        map will cover.
    normal_weight: int
        Height of the normals.
    path_res_factor: int
        Value that determines how much the resolution of the path will be increased. The higher this value, the more points will be
        created.
    Return:
    -----------
    idx_best_angles: list, int
        List containing the values of the best angles.
    """

    # SEE LATER
    # paths 1 and 2 are interpolated and their tangents are created
    path1_interp, tangents1 = smutil.increase_path_resolution(path1,  path_res_factor)
    path2_interp, tangents2 = smutil.increase_path_resolution(path2, path_res_factor)

    # LineString object is created by passing the interpolated path
    sh_path1_interp = geometry.LineString(path1_interp)
    sh_path2_interp = geometry.LineString(path2_interp)

    # normals do not point in the same direction as the original paths
    normals1 = smutil.get_normals(tangents1)
    normals2 = smutil.get_normals(tangents2)

    all_fitness = []
    idx_best_angles = []
    for idxm, pointm in enumerate(medial_path):
        normalm = medial_normals[idxm]
        candidate_line = np.array([pointm - reach * normalm, pointm, pointm + reach * normalm])
        sh_candidate_line = geometry.LineString(candidate_line)
        all_fitness.append([])
        for angle_idx, angle in enumerate(angles):
            sh_candidate_line_rotated = affinity.rotate(sh_candidate_line, angle)
            fitness = measure_fitness(sh_candidate_line_rotated, normalm, sh_path1_interp, normals1,
                                      sh_path2_interp, normals2, normal_weight)
            all_fitness[-1].append(fitness)

        idx_max = np.argmax(all_fitness[-1])
        if all_fitness[-1][idx_max] <= 0:
            idx_best_angles.append(None)
        else:
            idx_best_angles.append(idx_max)
            sh_candidate_line_rotated = affinity.rotate(sh_candidate_line, angles[idx_max])
            candidate_line_rotated = np.array(sh_candidate_line_rotated.coords)
    return idx_best_angles


def measure_fitness(sh_candidate_line, normalm, sh_path1, normals1, sh_path2, normals2, normal_weight):
    """Measures the fitness of the candidate line.

    Parameters:
    -----------
    sh_candidate_line: object, LineString
        Object of type shapely.geometry.linestring.LineString.
    normalm: ndarray, float
        Vector containing a pair of values.
    sh_path1: object, LineString
        Object of type shapely.geometry.linestring.LineString for path 1.
    normals1: ndarray, float
        Vector containing the normals of path 1.
    sh_path2: object, LineString
        Object of type shapely.geometry.linestring.LineString for path 2.
    normals2: ndarray, float
        Vector containing the normals of path 2.
    normal_weight: int
        Height of the normals.
    Return:
    -----------
    fitness: int
        Returns whether the chosen candidate line is the best option.
    """

    # tries to find the intersection point of the transverse paths
    sh_path1_point = find_envelop_cross_path_intersection(sh_candidate_line, sh_path1)
    sh_path2_point = find_envelop_cross_path_intersection(sh_candidate_line, sh_path2)

    # if there is no intersection, the fitness is -1, i.e., the candidate line has no intersection
    if sh_path1_point is None or sh_path2_point is None:
        fitness = -1
    else:
        # import pdb; pdb.set_trace()
        path1_point = np.array(sh_path1_point.coords)[0]
        path2_point = np.array(sh_path2_point.coords)[0]
        idx_path1_point = smutil.find_point_idx(sh_path1, path1_point)
        normal1 = normals1[idx_path1_point]
        idx_path2_point = smutil.find_point_idx(sh_path2, path2_point)
        normal2 = normals2[idx_path2_point]

        candidate_line_rotated = np.array(sh_candidate_line.coords)
        candidate_normal = candidate_line_rotated[-1] - candidate_line_rotated[0]
        candidate_normal = candidate_normal / np.sqrt(candidate_normal[0] ** 2 + candidate_normal[1] ** 2)
        medial_congruence = abs(np.dot(candidate_normal, normalm))
        path1_congruence = abs(np.dot(candidate_normal, normal1))
        path2_congruence = abs(np.dot(candidate_normal, normal2))
        fitness = normal_weight * medial_congruence + path1_congruence + path2_congruence

    return fitness

