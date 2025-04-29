import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
import matplotlib
matplotlib.use('TkAgg')

def get_building_matrix(center_x = 180787, center_y = 670770) -> np.ndarray:
    """
    returns a matrix representing the height of each building
    :param center_x:
    :param center_y:
    :return:
    """
    # --- Load shapefile ---
    gdf = gpd.read_file("city/Buildings.shp")

    height_field = 'maxheight'  # or whatever your height column is

    # --- Define the CENTER and SIZE manually ---
    grid_size_meters = 2000  # total size of the grid in meters (square)

    # --- Calculate the bounding box from center ---
    half_size = grid_size_meters / 2

    minx = center_x - half_size
    maxx = center_x + half_size
    miny = center_y - half_size
    maxy = center_y + half_size

    # --- Define raster/grid size ---
    pixel_size = 1  # 1 meter resolution

    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    # --- Clip buildings that are inside the box (optional but better) ---
    gdf = gdf.cx[minx:maxx, miny:maxy]

    # --- Rasterize ---
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[height_field]))

    building_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='float32'
    )

    building_raster = np.nan_to_num(building_raster, nan=0.0)
    half_size = int(half_size)
    building_raster[half_size - 57:half_size + 15, half_size-15:half_size+35] = 0
    return building_raster


def plot_buildings(building_raster):
    # --- Plot ---
    plt.figure(figsize=(10, 8))
    plt.imshow(building_raster, origin='upper', cmap='viridis')
    plt.colorbar(label='Building Height (meters)')
    plt.title('Building Heights Centered Around a Point')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True, color='white', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    plot_buildings(get_building_matrix())
