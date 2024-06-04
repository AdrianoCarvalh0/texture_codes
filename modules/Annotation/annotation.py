import json
from pathlib import Path
import dash                                                                                                                                                 
from dash import html,dcc
from jupyter_dash import JupyterDash
import plotly.express as px                                                                                                                                   
import numpy  as np                                                                                                                                           
from PIL import Image
from dash.dependencies import Input, Output

from skimage import draw
from scipy import ndimage
import re


def record_vector_file(array_list, filename):
  """Function to write an array to a JSON file"""
  array = [item.tolist() for item in array_list]
  json.dump(array, open(filename, 'w'), indent=2)


def path_to_floats(path):
    """Converts the SVG path to a numpy vector of coordinates, where each line becomes a float point (row, column)"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.array(indices_str, dtype=float)

def path_to_indices(path):
    """Converts the SVG path to a numpy vector of coordinates, where each line becomes an integer point (row, column)"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(np.int)


def path_to_mask(path, shape):
    """Converts the SVG path to a boolean array where all pixels included in the path
    are true and the other pixels are false."""
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def process_shape(shape):
    """Gets the coordinates of a shape drawn in the Dash app. Also returns the type of the shape
    as one of the following strings: rect, line, circle, closed_path and open_path."""

    if shape['type']=='rect' or shape['type']=='line' or shape['type']=='circle':

        x0, y0 = shape["x0"], shape["y0"]
        x1, y1 = shape["x1"], shape["y1"]
        if shape['type']!='line':  # Preserve point order if line
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

        if shape['type']=='rect':
            shape_type = 'rect'
            coords = (x0, y0, x1, y1)
        elif shape['type']=='line':
            shape_type = 'line'
            coords = (x0, y0, x1, y1)
        elif shape['type']=='circle':
            shape_type = 'circle'
            radius_h = (x1 - x0)/2
            radius_v = (y1 - y0)/2
            cx = x0 + radius_h
            cy = y0 + radius_v
            coords = (cx, cy, radius_h, radius_v)

    elif shape['type']=='path':
        if 'fillrule' in shape:
            shape_type = 'closed_path'
        else:
            shape_type = 'open_path'
        coords = path_to_floats(shape["path"])
    else:
        coords = None

    return shape_type, coords

def regexp_updated_shape(key):
    """Function to identify the index and attribute of a form that has been updated"""

    m = re.match(r'shapes\[(\d+)\]\.(\S+)', key)
    if m is None:
        return None
    else:
        shape_idx, att = m.groups()
        shape_idx = int(shape_idx)

    return shape_idx, att

def process_shape_update(relayout_data):
    """Returns the coordinates and type of the shape that was updated in the dash app."""

    shape = {}
    first_key = list(relayout_data.keys())[0]
    shape_idx, _ = regexp_updated_shape(first_key)
    shape_type = shapes[shape_idx]['type'] 
    if shape_type=='open_path' or shape_type=='closed_path':
        shape['type'] = 'path'
        path = relayout_data[first_key]
        shape['path'] = path
        if shape_type=='closed_path':
            shape['fillrule'] = 'evenodd'
    else:
        shape['type'] = shape_type
        for key in relayout_data:
            _, att = regexp_updated_shape(key)
            shape[att] = relayout_data[key]
    
    shape_type, coords = process_shape(shape)

    return shape_type, coords

def extract_intensities(img, shape_type, coords, flat=True):
    """Extracts the image intensities within the selected shape."""
   
    coords = np.rint(coords).astype(int)
    img_shape = img.shape

    if shape_type=='rect':
        x0, y0, x1, y1 = coords
        img_roi = img[y0:y1, x0:x1]
        if flat:
            img_roi = img_roi.ravel()
    elif shape_type=='line':
        x0, y0, x1, y1 = coords
        rr, cc = draw.line(y0, x0, y1, x1)
        img_roi = img[rr, cc]
    elif shape_type=='circle':
        c0, c1, radius_h, radius_v = coords
        rr, cc = draw.ellipse(c1, c0, radius_v, radius_h, shape=img_shape)
        if flat:
            img_roi = img[rr, cc]
        else:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[rr, cc] = True
            img_roi = img*mask
    elif shape_type=='closed_path' or shape_type=='open_path':
        cols, rows = coords.T
        if flat:
            rr, cc = draw.polygon(rows, cols, shape=img_shape)
            img_roi = img[rr, cc]
        else:
            mask = draw.polygon2mask(img_shape, zip(rows, cols))
            img_roi = img*mask

    return img_roi


def generate_dash(params):

    image = params['image']
    extension = params['extension']
    root_img = params['root_img']   # original images directory
    root_out = params['root_out']  # Path where the JSON file will be saved  
    list_array = params['list_array']  # array that will store the coordinates resulting from manual marking  
    
    # Annotation style for drawing on image
    annotation_style = {
        'line_color': 'red',
        'line_width': 2,
        'fillcolor': 'red',
        'opacity':1,
    }

    # Path where the JSON file will be saved
    file = f'{root_out}/{image}.json'

    # path of the image to be read
    path = f'{root_img}/{image}{extension}'

    # converting image to numpy array
    img = np.array(Image.open(path))

    # Initial image setup in Plotly Express
    fig = px.imshow(img, color_continuous_scale='gray', zmin=0, zmax=255,
            binary_string=True, binary_compression_level=0)
    fig.update_layout(dragmode="drawopenpath", coloraxis_showscale=False,
                newshape=annotation_style, height=1000, hovermode=False)

    # Configure tooltip
    fig.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> z: %{z} <br> color: %{color}")

    # Initial histogram
    fig_hist = px.histogram(img.ravel(), height=700)

    # Dash Settings
    config = {
        # Toolbar buttons
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
        'displayModeBar': True,
        'displaylogo': False,
    }

    # Dash Components
    dash_graph = dcc.Graph(id="graph-picture", figure=fig, config=config)

    # Instantiates an app object of the JupyterDash class
    app = JupyterDash(__name__)

    # Dash information, title, screen size
    app.layout = html.Div(
        [
            html.H3("Drag a rectangle to show the histogram of the ROI"),
            html.Div(
                [dash_graph,],
                style={"width": "60%", "display": "inline-block", "padding": "0 0"},
            ),
            html.Div(
                [dcc.Graph(id="histogram", figure=fig_hist),],
                style={"width": "40%", "display": "inline-block", "padding": "0 0"},
            ),
            dcc.Markdown("Characteristics of shapes"),
            html.Pre(id="annotations-data"),  # For debugging
        ]
    )

    shapes = []

    @app.callback(
        Output("histogram", "figure"),
        Output("annotations-data", "children"),
        Input("graph-picture", "relayoutData"),
        prevent_initial_call=True,
    )
    def on_new_annotation(relayout_data):

        if "shapes" in relayout_data and len(relayout_data['shapes'])>0:        
            #A shape has been drawn in the figure
            last_shape = relayout_data["shapes"][-1]
            shape_type, coords = process_shape(last_shape)

            # Adds shapes, storing type and coordinates
            shapes.append({'type':shape_type, 'coords':coords})

            # Add coordinates to an array
            list_array.append(coords)

            #Writes the array to a .json file
            record_vector_file(list_array, file)
            
            # Extracts the intensities to pass to the histogram
            img_roi = extract_intensities(img, shape_type, coords)

            return px.histogram(img_roi), json.dumps(relayout_data['shapes'], indent=2)

        elif any([re.match(r'shapes\[\d+\]\.', key) is not None for key in relayout_data]): 
            # Finds Strings like `shapes[3].x0` or `shapes[5].path`

            shape_type, coords = process_shape_update(relayout_data)
            img_roi = extract_intensities(img, shape_type, coords)

            return px.histogram(img_roi), json.dumps(relayout_data, indent=2)
        else:
            return (dash.no_update,)*2
        
    # Dash app main method. Creates a browser-accessible address for demarcating the vessel(s).
    app.run_server(mode='external')


