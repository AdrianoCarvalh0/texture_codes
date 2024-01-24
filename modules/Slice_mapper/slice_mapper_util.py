import math
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Voronoi
from shapely import geometry, ops as shops

def arc_length(path):
    """Calculate the accumulated arc length between two points

    Parameters:
    -----------
    path: ndarray
        List of points containing the path.

    Returns:
    -----------
    l: np.array, containing float values
        Array containing partially accumulated information of the sums of checked values.
        The last value in the list represents the total sum of all previously added differences.
    """

    # Calculate the difference between segments, square it, and take the square root.
    # It calculates the hypotenuse when pixels are diagonal; if they are horizontal, this value is 1.
    dl = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))

    # l is the cumulative sum of the values of dl, transformed into a list
    l = np.cumsum(dl).tolist()

    # Add the value zero to have the same number of points with arc lengths. Zero would be the difference from the first point.
    l = np.array([0] + l)

    # Return the values of all arc lengths
    return l

def interpolate(path, delta_eval=2., smoothing=0.1, k=3, return_params=False):
    """Interpolate a list of points

    Parameters:
    -----------
    path: ndarray
        List of points containing the path.
    delta_eval: float
        Variable responsible for creating points between segments.
    smoothing: float
        Degree of smoothing.
    k: int
        Parameter for cubic interpolation.
    return_params: boolean
        Parameter set to false. If true, it returns additional values.

    Returns:
    -----------
    path_interp: ndarray, containing float values
        Array containing partially accumulated information of the sums of checked values.
        The last value in the list represents the total sum of all previously added differences.
    tangent: ndarray
        Vector containing a list of tangents.
    tck: list
        List containing curve characteristics.
    u: ndarray
        Contains values between the interval 0-1 and with num_points indices
    """

    # Convert the path to np.array
    path = np.array(path)

    # Absorb the accumulated arc length. All positions of the arc length.
    l = arc_length(path)

    # Number of points absorbs the size of the path
    num_points = len(path)

    # tck curve characteristics
    # Use splprep from scipy to perform interpolation
    (tck, u), fp, ier, msg = splprep(path.T, s=smoothing * num_points, k=k, full_output=True)

    # The l in the last position is the accumulated value of all sums of arc length
    # Delta_eval_norm is the division between delta_eval and the total length of the arc
    delta_eval_norm = delta_eval / l[-1]

    # Eval_points ==> variation between 0, interval, of delta_eval_norm in delta_eval_norm
    eval_points = np.arange(0, 1 + 0.75 * delta_eval_norm, delta_eval_norm)

    # Interpolated points
    x_interp, y_interp = splev(eval_points, tck, ext=3)

    # Derivatives of interpolated points, der=1, gives the derivative instead of the point
    dx_interp, dy_interp = splev(eval_points, tck, der=1, ext=3)

    # .T inverts instead of having two rows and num_points columns, it becomes num_points rows and two columns
    path_interp = np.array([x_interp, y_interp]).T

    # Transposed tangent vectors through the derivatives of x and y
    tangent = np.array([dx_interp, dy_interp]).T

    # Normalize tangents to have a maximum size of 1 and not worry about size
    t_norm = np.sqrt(np.sum(tangent ** 2, axis=1))
    tangent = tangent / t_norm[None].T

    if return_params:
        return path_interp, tangent, tck, u
    else:
        return path_interp, tangent



def two_stage_interpolate(path, delta_eval=2., smoothing=0.1, k=3):
    """Interpolate the path in two stages. First, a linear interpolation is applied to generate
    intermediate points. Then, a cubic interpolation is applied. This is useful
    because cubic interpolation ensures that the spline passes close to the original points
    on the path but may be far from the original curve between two original points. By doing
    a linear interpolation first followed by a cubic one, the resulting spline cannot be
    too far from the original path.

    Parameters:
    -----------
    path: ndarray
        List of points containing the path to be interpolated.
    delta_eval: float
        The interval to evaluate the interpolation.
    smoothing: float
        Smoothing factor. 0 means the spline will pass through all points linearly interpolated.
    k: int
        The degree of the second interpolation - which in this case is cubic.

    Returns:
    -----------
    path_interp: ndarray, float
        Interpolated path first linearly and then cubically.
    tangent: ndarray, float
        ndarray of tangents
    """

    path_interp_linear, _ = interpolate(path, delta_eval=delta_eval, smoothing=0, k=1)
    path_interp, tangent = interpolate(path_interp_linear, delta_eval=delta_eval, smoothing=smoothing, k=k)

    return path_interp, tangent


def get_normals(tangents):
    """Get normal vectors based on a list of tangent vectors

    Parameters:
    -----------
    tangents: ndarray, float
        ndarray of tangents

    Returns:
    -----------
    normals: ndarray, float
        ndarray of normals
    """
    # Create a matrix of zeros with two columns and the size of the tangents vector
    normals = np.zeros((len(tangents), 2))

    # Loop that retrieves the indices and values of the tangents, idx ==> index and t ==> values
    for idx, t in enumerate(tangents):

        # tx and ty absorb the values of the tangents
        tx, ty = t

        # If ty has a value very close to zero
        if ty < 1e-3:
            n2 = 1
            n1 = -ty * n2 / tx
        else:
            n1 = 1
            n2 = -tx * n1 / ty

        # Applying normalization
        norm = np.sqrt(n1 ** 2 + n2 ** 2)
        n = np.array([n1 / norm, n2 / norm])

        # np.cross ==> returns the cross product of two vectors
        # np.sign ==> returns -1 if x<0, 0 if x==0, and 1 if x>0
        orient = np.sign(np.cross(t, n))
        if idx > 0:
            if orient != prev_orient:
                # If the orientation of the vector is different from the previous vector, there is a change in orientation
                n *= -1
                orient *= -1
        prev_orient = orient
        normals[idx] = n

    return normals



def dist(p1, p2):
    """Calculate the Euclidean distance between two points

    Parameters:
    -----------
    p1: array
        Position 1 of a vector
    p2: array
        Position 2 of a vector

    Returns:
    -----------
    The calculation of the distance
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def medial_voronoi_ridges(path1, path2):
    """Extraction of the medial edges of Voronoi between path1 and path2. Voronoi diagrams can be
    used to represent the medial path of a tubular structure.

    Parameters:
    -----------
    path1: array
        Vector 1
    path2: array
        Vector 2
    Returns:
    -----------
    vor: Voronoi object
        Voronoi object containing information about the region
    idx_medial_vertices: ndarray
        Indices of medial vertices
    point_relation: ndarray
        Relationship points between one medial edge and another
    """

    # Create a new array concatenated between the paths along the rows
    all_points = np.concatenate((path1, path2), axis=0)

    # Sort all points
    all_points_ordered = np.concatenate((path1, path2[::-1]), axis=0)

    # Create the voronoi object, passing all concatenated points along the rows
    vor = Voronoi(all_points)

    # Number of points in path1
    num_points_path1 = len(path1)

    # Create a tubular region passing all ordered points
    tube_region = geometry.Polygon(all_points_ordered)

    idx_internal_vertices = set()
    # Get Voronoi vertices inside the tube
    for idx_vertex, vertex in enumerate(vor.vertices):
        if tube_region.contains(geometry.Point(vertex)):
            idx_internal_vertices.add(idx_vertex)

    idx_medial_vertices = []
    point_relation = []
    for idx, ridge in enumerate(vor.ridge_points):

        first_is_path1 = True if ridge[0] < num_points_path1 else False
        second_is_path1 = True if ridge[1] < num_points_path1 else False
        if (first_is_path1 + second_is_path1) == 1:
            # Check if the medial edge is between a point in path1 and another in path2
            idx_ridge_vertices = vor.ridge_vertices[idx]
            if idx_ridge_vertices[0] in idx_internal_vertices and idx_ridge_vertices[1] in idx_internal_vertices:
                # Be careful that -1 in idx_ridge_vertices is not in the terminal points
                idx_medial_vertices.append(idx_ridge_vertices)
                if ridge[0] < num_points_path1:
                    point_relation.append((ridge[0], ridge[1]))
                else:
                    point_relation.append((ridge[1], ridge[0]))

    idx_medial_vertices = np.array(idx_medial_vertices)
    point_relation = np.array(point_relation)

    return vor, idx_medial_vertices, point_relation


def order_ridge_vertices(idx_vertices):
    """Sorts the vertices of Voronoi medial edges. A list of Voronoi medial edges, which are not ordered, is passed
    as a parameter, and when the function is executed, these vertices defining a path are ordered.

    Parameters:
    -----------
    idx_vertices: ndarray, int
        Indices of vertices

    Returns:
    -----------
    ordered_vertices: ndarray, int
        Ordered vertices
    """

    idx_vertices = list(map(tuple, idx_vertices))
    vertice_ridge_map = {}
    last_vertex = -1
    for idx_ridge, (idx_v1, idx_v2) in enumerate(idx_vertices):
        if idx_v1 in vertice_ridge_map:
            vertice_ridge_map[idx_v1].append(idx_ridge)
        else:
            vertice_ridge_map[idx_v1] = [idx_ridge]

        if idx_v2 in vertice_ridge_map:
            vertice_ridge_map[idx_v2].append(idx_ridge)
        else:
            vertice_ridge_map[idx_v2] = [idx_ridge]

    for idx_vertex, indices_ridge in vertice_ridge_map.items():
        if len(indices_ridge) == 1:
            idx_first_vertex = idx_vertex
            break

    ordered_vertices = [idx_first_vertex]
    idx_ridge = vertice_ridge_map[idx_first_vertex][0]
    idx_v1, idx_v2 = idx_vertices[idx_ridge]
    if idx_v1 == idx_first_vertex:
        idx_vertex = idx_v2
    else:
        idx_vertex = idx_v1
    ordered_vertices.append(idx_vertex)
    prev_idx_ridge = idx_ridge
    prev_idx_vertex = idx_vertex
    while True:
        indices_ridge = vertice_ridge_map[idx_vertex]
        if len(indices_ridge) == 1:
            break
        if indices_ridge[0] == prev_idx_ridge:
            idx_ridge = indices_ridge[1]
        else:
            idx_ridge = indices_ridge[0]
        idx_v1, idx_v2 = idx_vertices[idx_ridge]
        if idx_v1 == prev_idx_vertex:
            idx_vertex = idx_v2
        else:
            idx_vertex = idx_v1

        ordered_vertices.append(idx_vertex)
        prev_idx_ridge = idx_ridge
        prev_idx_vertex = idx_vertex

    return ordered_vertices


def invert_if_opposite(path1, path2):
    """Inverts path2 if path1 and path2 are marked in opposite directions.
    This happens when we mark the vessels from right to left and the other from left to right, or vice versa.

    Parameters:
    -----------
    path1: ndarray, float
        Path 1 vector
    path2: ndarray, float
        Path 2 vector

    Returns:
    -----------
    path2: ndarray, float
        Inverted or not inverted path2, according to the checks
    """

    # Check the minimum size between the two paths
    min_size = min([len(path1), len(path2)])

    # Check distances between points. If the inverted distance is greater, path2 will be inverted
    avg_dist = np.sum(np.sqrt(np.sum((path1[:min_size] - path2[:min_size]) ** 2, axis=1)))
    avg_dist_inv = np.sum(np.sqrt(np.sum((path1[:min_size] - path2[::-1][:min_size]) ** 2, axis=1)))
    if avg_dist_inv < avg_dist:
        # Invert the path2 vector
        path2 = path2[::-1]

    # Return path2
    return path2


def increase_path_resolution(path, res_factor):
    """Increases the resolution of a given path by applying a factor.

    Parameters:
    -----------
    path: ndarray, float
        Path vector
    res_factor: int
       Value determining how much the path resolution will be increased. The higher this value, the more points will be created.

    Returns:
    -----------
    path_interp: ndarray, float
        Interpolated path
    tangents: ndarray, float
        Tangent vector created from the path, absorbing the values contained in the tangents in x and y
    """

    # x absorbs the values from the path rows
    # y absorbs the values from the path columns
    x, y = path.T

    # number of points absorbs the size of the path
    num_points = len(path)

    indices = list(range(num_points))
    # Define the parametric variable making sure it passes through every original point
    tck, _ = splprep(path.T, u=indices, s=0, k=3)

    # eval_points is a variation of the value num_points*res_factor - (res_factor-1), ranging from 0 to num_points -1
    eval_points = np.linspace(0, num_points - 1, num_points * res_factor - (res_factor - 1))

    # interpolation of x and y
    x_interp, y_interp = splev(eval_points, tck, der=0)

    # creation of the tangent in x and y
    x_tangents, y_tangents = splev(eval_points, tck, der=1)

    # creation of the interpolated path
    path_interp = np.array([x_interp, y_interp]).T

    # creation of the tangents
    tangents = np.array([x_tangents, y_tangents]).T

    return path_interp, tangents


def find_point_idx(sh_path, point):
    """Finds the index of the point in sh_path.

     Parameters:
    -----------
    sh_path: ndarray, float
        Path vector
    point: int
        Index of the point

    Returns:
    -----------
        Minimum distances between the path and the point
    """

    # application of Euclidean distance to find the minimum distances between two points
    dists = np.sqrt((sh_path.xy[0] - point[0]) ** 2 + (sh_path.xy[1] - point[1]) ** 2)

    # return the minimum distances
    return np.argmin(dists)

