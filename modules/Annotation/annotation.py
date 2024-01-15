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

def gravar_array_arquivo(array_list, filename):
  """Função para gravar um array em um arquivo JSON """
  lista2 = [item.tolist() for item in array_list]
  json.dump(lista2, open(filename, 'w'), indent=2)


def path_to_floats(path):
    """Converte o caminho SVG para um vetor numpy de coordenadas, onde cada linha se torne um ponto float (linha, coluna)  
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.array(indices_str, dtype=float)

def path_to_indices(path):
    """Converte o caminho SVG para um vetor numpy de coordenadas, onde cada linha se torne um ponto inteiro (linha, coluna)    
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(np.int)


def path_to_mask(path, shape):
    """Converte o caminho SVG em uma matriz booleana onde todos os pixels incluídos no caminho
    são verdadeiros e os outros pixels são falsos.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

def process_shape(shape):
    """Obtém as coordenadas de uma forma desenhada no aplicativo Dash. Também retorna o tipo da forma
    como uma das seguintes strings: rect, line, circle, closed_path e open_path."""

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
    """Função para identificar o índice e o atributo de uma forma que foi atualizada"""

    m = re.match(r'shapes\[(\d+)\]\.(\S+)', key)
    if m is None:
        return None
    else:
        shape_idx, att = m.groups()
        shape_idx = int(shape_idx)

    return shape_idx, att

def process_shape_update(relayout_data):
    """Retorna as coordenadas e o tipo da forma que foi atualizada na dash app."""

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
    """Extrai as intensidades da imagen dentro da forma selecionada."""

    #coords = np.rint(coords).astype(np.int)
    #Alterei esta parte para int
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

# Estilo de anotação para desenho na imagem
annotation_style = {
    'line_color': 'red',
    'line_width': 2,
    'fillcolor': 'red',
    'opacity':1,
}

imag = 'Experiment #1 (adults set #1)_20x_batch1 - Superfical layers@64-Image 4-20X'

#root_dir linux
root_dir ="/home/adriano/projeto_mestrado/modules"

#root_dir windows
#root_dir = Path(r"C:\Users\adria\Documents\Mestrado\texture_codes\modules")

# Caminho onde vai ser gravado o arquivo JSON
arquivo = f'{root_dir}/Vetores_Extraidos_json/novos/{imag}.json'

# caminho da imagem a ser lida
path = f'{root_dir}/Imagens/vessel_data/images/{imag}.tiff'

# conversão da imagem em array numpy
img = np.array(Image.open(path))

# Configuração da imagem inicial no Plotly Express
fig = px.imshow(img, color_continuous_scale='gray', zmin=0, zmax=100,
        binary_string=True, binary_compression_level=0)
fig.update_layout(dragmode="drawopenpath", coloraxis_showscale=False,
            newshape=annotation_style, height=1000, hovermode=False)

# Configure tooltip
fig.update_traces(hovertemplate="x: %{x} <br> y: %{y} <br> z: %{z} <br> color: %{color}")

# Histograma inicial
fig_hist = px.histogram(img.ravel(), height=700)

# Configurações do Dash
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

# Componentes do Dash
dash_graph = dcc.Graph(id="graph-picture", figure=fig, config=config)

# Instancia um objeto app da classe JupyterDash
app = JupyterDash(__name__)

# Informações da Dash, título, tamanho da tela
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
        # Uma forma foi desenhada na figura
        last_shape = relayout_data["shapes"][-1]
        shape_type, coords = process_shape(last_shape)

        # Adiciona as formas, armazenando o tipo e as coordenadas
        shapes.append({'type':shape_type, 'coords':coords})

        # Adiciona as coordenadas em um array
        lista.append(coords)

        # Grava o array em um arquivo .json
        gravar_array_arquivo(lista, arquivo)
        
        # Extrai as intensidades para passar para o histograma
        img_roi = extract_intensities(img, shape_type, coords)

        return px.histogram(img_roi), json.dumps(relayout_data['shapes'], indent=2)

    elif any([re.match(r'shapes\[\d+\]\.', key) is not None for key in relayout_data]): 
        # Encontra Strings tipo 'shapes[3].x0` or `shapes[5].path`

        shape_type, coords = process_shape_update(relayout_data)
        img_roi = extract_intensities(img, shape_type, coords)

        return px.histogram(img_roi), json.dumps(relayout_data, indent=2)
    else:
        return (dash.no_update,)*2

# Método principal do Dash app. Cria um endereço acessível em navegador para a demarcação do(s) vaso(s).
if __name__ == "__main__":
    lista = []
    app.run_server(mode='external')