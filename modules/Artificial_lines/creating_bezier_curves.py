import numpy as np
from matplotlib import pyplot as plt
import sys

# Linux
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
from modules.Utils import functions  # Importing functions from the specified module
from pathlib import Path

# Linux
# sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
# root_dir = f"/home/adriano/projeto_mestrado/modules"

# Windows
sys.path.insert(0, r"C:\Users\adria\Documents\Masters\texture_codes\modules")
root_dir = Path(r"C:\Users\adria\Documents\Masters\texture_codes\modules")


def bezier(points, precision):
    """Function that creates Bezier curves

    Parameters:
    -----------
    points: ndarray
        array containing control points
    precision: int
        number of points to be created between the initial and final points
    Returns:
    -----------
    B: ndarray
        Stores the accumulated values of control points weighted by Bernstein coefficients.
    """
    # generate a sequence of numbers between 0 and 1, depending on the precision value
    ts = np.linspace(0, 1, precision)
    # create a matrix with two columns and the number of rows depending on the size of ts (filled with zeros), with float type
    result = np.zeros((len(ts), 2), dtype=np.float64)
    n = len(points) - 1

    for idx, t in enumerate(ts):
        for i in range(n + 1):
            # the binomial coefficient is used to weigh the influence of each control point on the final curve
            bin_coef = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))

            # Bernstein polynomial terms
            Pin = bin_coef * (1 - t) ** (n - i) * t ** i

            # each index of B receives the multiplication of the result of weighing each coefficient
            result[idx] += Pin * points[i]
    return result


def create_points(num_points):
    """Function that creates random points (coordinates) in the plane, based on the desired number of points (num_points)
    and a specified Euclidean distance range.

    Parameters:
    -----------
    num_points: int
        value determining the number of random points to be generated
    Returns:
    -----------
    points: ndarray float
        Stores the points.
    distance: float
        stores the value of the distance between the initial and final points
    """
    # element used to avoid exceeding the boundary
    padding = 20

    n_rows = 1504
    n_columns = 1776

    points = []
    while len(points) < num_points:
        # random selection of points
        p1x = np.random.randint(0, n_columns - padding)
        p1y = np.random.randint(0, n_rows - padding)
        p2x = np.random.randint(0, n_columns - padding)
        p2y = np.random.randint(0, n_rows - padding)

        # Euclidean distance between initial and final coordinates
        distance = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)

        # only add elements to the points array whose distance is greater than 300 pixels and less than 500
        if 500 < distance < 1300:
            p1 = np.array((p1x, p1y))
            p2 = np.array((p2x, p2y))

            # stack the points
            p = np.vstack((p1, p2))
            points.append(p)

    # return the points created along the path and the distance.
    # The distance element will serve as the basis for setting the length (number of columns) of the transformed maps.
    return points, distance


def create_array_curves(points):
    array_curves = []
    ps = points[0][0]  # initial point
    pe = points[0][1]  # final point
    dx = pe[0] - ps[0]
    dy = pe[1] - ps[1]
    distance = np.sqrt((pe[0] - ps[0]) ** 2 + (pe[1] - ps[1]) ** 2)
    normal_se = np.array((-dy, dx)) / distance  # or (dy, -dx) --> vector normal to (pe-ps)
    max_vd = 500  # maximum distance where control points will be randomly drawn. Example: 1 generates straight lines.
    n_points = 6  # number of control points between pe and ps. The higher this number, the more curves are generated.

    control_points = []
    hds = np.linspace(0.2, 0.8, n_points)  # makes the control points equidistant relative to (pe-ps)

    for j in range(n_points):
        control_point = ((pe - ps) * hds[j])  # setting the horizontal distances this way gives a more natural look
        control_point += (normal_se * np.random.uniform(low=-1, high=1) * max_vd)
        control_points.append(control_point + ps)

    control_points.insert(0, ps)
    control_points.append(pe)
    curve = bezier(control_points, precision=100)
    array_curves.append(curve)

    return array_curves


if __name__ == "__main__":

    bezier_paths = f'{root_dir}/Artificial_Lines/bezier_paths/'

    array_curves = []
    for i in range(5):
        points, distance = create_points(25)
        curve = create_array_curves(points)
        data_dict = {
            "curve": curve,
            "distance": distance
        }
        curve.append(distance)
        functions.save_dict_to_file(data_dict, f'{bezier_paths}/img_savedata_{i}.json')
        array_curves.append(curve)
