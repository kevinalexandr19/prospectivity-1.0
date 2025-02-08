# PROSPECTIVITY
# VERSION 1.0

import pandas as pd
import rioxarray as riox
import numpy as np
import pyvista as pv
from tqdm.notebook import tqdm
from numba import jit
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

def get_colormap():
    colors = np.zeros((256, 3))

    # Red
    colors[:32, 0] = np.linspace(0.07, 0.03, 32)
    colors[32:64, 0] = np.linspace(0.03, 0.1, 32)
    colors[64:96, 0] = np.linspace(0.1, 0.17, 32)
    colors[96:128, 0] = np.linspace(0.17, 0.52, 32)
    colors[128:160, 0] = np.linspace(0.52, 0.9, 32)
    colors[160:192, 0] = np.linspace(0.9, 1, 32)
    colors[192:224, 0] = np.linspace(1, 1, 32)
    colors[224:, 0] = np.linspace(1, 0.74, 32)
    
    # Green
    colors[:32, 1] = np.linspace(0.08, 0.07, 32)
    colors[32:64, 1] = np.linspace(0.07, 0.28, 32)
    colors[64:96, 1] = np.linspace(0.28, 0.59, 32)
    colors[96:128, 1] = np.linspace(0.59, 0.74, 32)
    colors[128:160, 1] = np.linspace(0.74,0.87, 32)
    colors[160:192, 1] = np.linspace(0.87, 0.68, 32)
    colors[192:224, 1] = np.linspace(0.68, 0.34, 32)
    colors[224:, 1] = np.linspace(0.34, 0.08, 32)
    
    # BLue
    colors[:32, 2] = np.linspace(0.28, 0.74, 32)
    colors[32:64, 2] = np.linspace(0.74, 0.93, 32)
    colors[64:96, 2] = np.linspace(0.93, 0.37, 32)
    colors[96:128, 2] = np.linspace(0.37, 0, 32)
    colors[128:160, 2] = np.linspace(0, 0, 32)
    colors[160:192, 2] = np.linspace(0, 0.08, 32)
    colors[192:224, 2] = np.linspace(0.08, 0.04, 32)
    colors[224:, 2] = np.linspace(0.04, 0.08, 32)
    
    cm = LinearSegmentedColormap.from_list("raster_cmap", colors, N=256)
    return cm


def read_raster(file):
    """Open raster file and replace empty values with NaN."""
    # Abrir archivo
    raster = riox.open_rasterio(file)
    # Reemplazar vacíos por np.nan
    nans = (raster.values == raster.rio.nodata)
    raster.values = raster.values.astype(float)
    raster.values[nans] = np.nan
    return raster


def get_index(raster, x, y):
    """Extract the nearest raster column and row location from an xy point."""
    # Usar sel con método "nearest" para obtener el píxel más cercano al punto
    nearest_pixel = raster.sel(x=x, y=y, method="nearest")
    # Obtener la fila y columna correspondiente al píxel más cercano
    col = (raster.coords["x"].values == nearest_pixel.coords["x"].values).argmax().astype(int)
    row = (raster.coords["y"].values == nearest_pixel.coords["y"].values).argmax().astype(int)
    return (col, row)


def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    

# def generate_table_crops(table, array, size, wtscol, typecol):
#     """Crop square matrices from an RGB array, with an specific size and using xy points as the centroids.
#     Weights and type columns are also added to the result."""
#     assert size % 2 == 1, "Parameter size must be an odd number."
#     shape = array.shape
    
#     crops = []
#     locations = set()
#     for _, data in table.iterrows():
#         col, row = data["col"], data["row"]
#         start_index = size // 2
#         col_start = col - start_index
#         row_start = row - start_index
#         # Verifica que los puntos se encuentren en un margen dentro del raster
#         if any([(row_start < 0), (col_start < 0), (row_start + size > shape[0]), (col_start + size > shape[1])]):
#             continue
#         # Extrae un slice del raster centrado en el punto
#         slice = array[row_start:(row_start+size), col_start:(col_start+size), :]
#         # Almacena el slice en la lista si no contiene nans
#         if (np.isnan(slice).sum() == 0) and ((col, row) not in locations):
#             locations.add((col, row))
#             crops.append(dict(index=(col, row), wts=data[wtscol], type=data[typecol], slice=slice))

#     return crops

def generate_reference_tiles(table, array, size, index=None):
    """Crop square matrices (tiles) from an RGB array, based on a table of occurrences,
    using an specific size and xy points as the centroids."""
    assert size % 2 == 1, "Parameter size must be an odd number."
    
    shape = array.shape
    tiles = []
    locations = set()
    
    # Loop that iterates for every row in table of occurrences
    for _, data in table.iterrows():
        # Get position of the tile
        col, row = int(data["col"]), int(data["row"])
        start_index = size // 2
        col_start = col - start_index
        row_start = row - start_index
        
        # Verify that tile is inside the raster
        if any([(row_start < 0), (col_start < 0), (row_start + size > shape[0]), (col_start + size > shape[1])]):
            continue
            
        # Extract tile from data (row, col, channels)
        tile = array[row_start:(row_start+size), col_start:(col_start+size), :]
        
        # Store the tile if it does not contain NaN values
        if (np.isnan(tile).sum() == 0) and ((col, row) not in locations):
            locations.add((col, row))
            tiles.append(dict(index=(col, row), array=tile))

    if index:
        tiles = [tile for tile in tiles if tile["index"] in index]

    return tiles


def generate_rgb_tiles(array, index, size):
    """Crop all available square matrices (tiles) from an RGB array, using an specific size."""
    assert size % 2 == 1, "Parameter size must be an odd number."
    
    rows, cols = array.shape[:2]
    half_size = size // 2
    tiles = []

    # Loop through all available tiles in the image
    for row in range(rows // size):
        for col in range(cols // size):
            # Get initial position of tile
            col_start = index[0] + (col * size)
            row_start = index[1] + (row * size)
            
            # Extract tile from data (row, col, channels)
            tile = array[row_start:(row_start+size), col_start:(col_start+size), :]
            
            # Store the tile if it does not contain NaN values
            if (np.isnan(tile).sum() == 0):
                tiles.append(dict(index=(col_start + half_size, row_start + half_size), array=tile))

    return tiles


def get_embedding(image, base_model):
    """Get embedding vector from an RGB image."""       
    # Multiplied by 255 to scale values back to [0, 255]
    image_preprocessed = tf.keras.applications.efficientnet.preprocess_input(image * 255)
    image_preprocessed = tf.expand_dims(image_preprocessed, 0)
    
    # Obtain vector embedding from the image
    embedding_vector = base_model.predict(image_preprocessed, verbose=0)
    return embedding_vector.reshape(-1)


@jit(nopython=True)
def cosine_similarity(A, B):
    """Cosine similarity between two vectors."""
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))



####### FOR PYVISTA
def generate_mesh(raster):
    """Create a mesh from a raster for 3D visualization in PyVista."""
    # Create grid x, y, z
    xx, yy = np.meshgrid(raster.x, raster.y)
    zz = raster.values.reshape(xx.shape)  # will make z-comp the values in the file
    # Create mesh
    mesh = pv.StructuredGrid(xx, yy, zz)
    mesh["data"] = raster.values.ravel(order="F")
    return mesh


def generate_crop_raster(raster, crops, size):
    """Create a new raster with the same metadata but filled only with the square matrices, the rest is filled with NaN."""
    new_raster = raster.copy(data=np.full_like(raster.data, fill_value=np.nan))
    
    # Add each patch to the new raster
    for crop in crops:
        col, row = crop["index"]
        start_index = size // 2
        col_start = col - start_index
        row_start = row - start_index 
        # Asignar los valores del parche al raster nuevo
        new_raster[:, row_start:(row_start+size), col_start:(col_start+size)] = crop["slice"]

    return new_raster

