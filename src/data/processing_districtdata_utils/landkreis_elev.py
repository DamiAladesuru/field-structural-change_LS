# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "lower_saxony_fisc": # lower_saxony_fisc or workspace
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

from src.data.processing_fielddata_utils import field_dataset as fd
from src.data.processing_districtdata_utils import landkreis_desc as ld

# %%
###########################################
# Landkreis level
landkreisdf = ld.create_fulllandkreisdf(return_mode='without_small', ref_year=2016)
landkreisdf_cl = ld.clean_landkreisdf(landkreisdf)
_, landkreis_yearly_cl = ld.silence_prints(ld.desc_landkreis, landkreisdf_cl, ref_year=2016)

# %%
# make landkreisgdf_cl because we need to spatial join with climate and elevation data
geom, landkreisgdf_cl = ld.to_gdf(landkreisdf_cl)
#%%
geom = geom[geom['LANDKREIS'].isin(landkreisgdf_cl['LANDKREIS'])].reset_index(drop=True)


# %% load gld
gld_no_geom = fd.process_nogeom_data()

###################################################################
# Obtain main cultivated crop or kulturart FOR EACH landkreis
###################################################################
# %% Sum area_ha across years by LANDKREIS and kulturart
area_sum_dict = {}

for year, df in gld_no_geom.items():
    if 'area_ha' in df.columns:
        area_sum = (
            df.groupby(['LANDKREIS', 'kulturart', 'Gruppe'], as_index=False)['area_ha']
              .sum()
              .rename(columns={'area_ha': 'total_area_ha'})
        )
        # assign the year
        area_sum['year'] = year
        # save to dict
        area_sum_dict[year] = area_sum
    else:
        print(f"⚠️ Year {year} has no 'area_ha' column, skipping.")

# put all years' into one dataframe
area_sum_allyears = pd.concat(area_sum_dict.values(), ignore_index=True)
area_sum = (area_sum_allyears.groupby(['LANDKREIS', 'kulturart', 'Gruppe'], as_index=False)['total_area_ha']
            .sum())

# For each LANDKREIS, find the kulturart with the maximum total_area_ha
idx = area_sum.groupby('LANDKREIS')['total_area_ha'].idxmax()

dominant_kulturart = area_sum.loc[idx].reset_index(drop=True)
# dominant_kulturart has columns: LANDKREIS, kulturart, Gruppe, total_area_ha

landkreis_crop = landkreisgdf_cl.merge(
    dominant_kulturart[['LANDKREIS', 'kulturart', 'Gruppe', 'total_area_ha']],
    on='LANDKREIS',
    how='left'
)
landkreis_crop.info()

#%% examine data in maps
def plot_landkreis_with_attribute(landkreis_gdf, attr_col, title):
    fig, ax = plt.subplots(figsize=(12, 12))
    landkreis_gdf.plot(column=attr_col, ax=ax, cmap='Dark2', legend=True, edgecolor='gray', linewidth=0.3)
    ax.set_title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
plot_landkreis_with_attribute(landkreis_crop, "Gruppe", "LANDKREIS - Dominant CropGroup")

# %%
# column renames
landkreisgdf_cl = landkreis_crop.rename(columns={"kulturart": "main_crop",
                                                        "Gruppe": "main_crop_group",
                                                        "total_area_ha": "main_crop_totarea"})

# %%
# gdf contains unique landkreis, their landkreis, predominant climate region,
# predominant crop type and associated group

# Check for missing values in the merged DataFrame
missing_values = landkreisgdf_cl.isnull().sum()
print("Missing values in the merged DataFrame:")
print(missing_values[missing_values > 0])

# %%
# check the first and last columns of landkreisgdf_cl
print("First columns:")
print(landkreisgdf_cl.columns[:10])
print("Last columns:")
print(landkreisgdf_cl.columns[-10:])
       
###########################################
# add mean elevation data
###########################################
# %% 1. checkout raster data
raster_file_path = data_main_path+"/raw/lsax_elevation.tif"
with rasterio.open(raster_file_path) as src:
    print(f"Raster: {raster_file_path}")
    print(f"- CRS: {src.crs}")
    print(f"- Dimensions: {src.width} x {src.height}")
    print(f"- Bands: {src.count}")
    print(f"- NoData: {src.nodata}")

    for i in range(1, src.count + 1):
        band = src.read(i)
        print(f"\nBand {i} stats:")

        # Apply NoData mask
        if src.nodata is not None:
            band = np.ma.masked_equal(band, src.nodata)
        else:
            band = np.ma.masked_invalid(band)  # in case of NaN values

        # Check if band is not completely masked
        if band.count() > 0:
            print(f"  Min:  {band.min()}")
            print(f"  Max:  {band.max()}")
            print(f"  Mean: {band.mean():.2f}")
            print("Shape of band:", band.shape)
            print("First few values:", band[:5, :5])
        else:
            print("  All values are NoData or masked.")

with rasterio.open(raster_file_path) as src:
    elevation_data_m = src.read(1)
    raster_crs = src.crs
    print(raster_crs)

elevation_data = elevation_data_m / 1000

# Plot the raster data using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(elevation_data, cmap='viridis')  # You can change the colormap (cmap)
plt.colorbar(label="Elevation (km)")  # Add a color bar to show elevation scale
plt.title("Raster Elevation Data")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()

# %% 2. create landkreisgdf by removing duplicates so as to keep only one year's unique landkreis
# and reproject raster data
landkreisgdf = landkreisgdf_cl[['LANDKREIS', 'geometry']].drop_duplicates('LANDKREIS')

target_crs = landkreisgdf.crs

# Input and output paths
src_path = data_main_path+"/raw/lsax_elevation.tif"
dst_path = data_main_path+"/raw/lsax_elevation_reprojected.tif"

with rasterio.open(src_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds)
    
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(dst_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear)

# %% 3. check that all of landkreisgdf geometries is covered by elevation bounds
# Load the reprojected raster to get its bounds
with rasterio.open(dst_path) as src:
    raster_bounds = box(*src.bounds)
    raster_crs = src.crs

# Create a GeoDataFrame for the raster extent
raster_extent_gdf = gpd.GeoDataFrame({'geometry': [raster_bounds]}, crs=raster_crs)

# Check which landkreis intersect the raster
intersections = gpd.sjoin(landkreisgdf, raster_extent_gdf, predicate='intersects', how='left')
covered_cells = intersections['index_right'].notna().sum()
print(f"landkreis overlapping raster: {covered_cells} / {len(landkreisgdf)}")

# Optional: remove non-intersecting geometries
landkreisgdf = intersections[intersections['index_right'].notna()].drop(columns='index_right')


# %% 4. compute mean elevation for every geometry row in landkreisgdf
# Compute mean elevation per landkreis
elevation_stats = zonal_stats(
    landkreisgdf,
    dst_path,
    stats=["mean"],
    nodata=-32768,  # <- replace with actual nodata if different
    geojson_out=True
)


# Build GeoDataFrame from results
landkreisgdf['mean_elevation_m'] = [feat['properties']['mean'] for feat in elevation_stats]
landkreisgdf['mean_elevation'] = landkreisgdf['mean_elevation_m'] / 1000
landkreisgdf.info()


# %% 5.
missing = landkreisgdf[landkreisgdf['mean_elevation'].isna()]
print(f"⚠️ Rows with missing elevation: {len(missing)}")

missing_landkreis = landkreisgdf[landkreisgdf['mean_elevation'].isna()]['LANDKREIS'].unique()
print("LANDKREIS with missing elev:")
print(missing_landkreis)


# %% 6. merge back to yearly data
landkreisgdf_cl_new = landkreisgdf_cl.merge(landkreisgdf[['LANDKREIS', 'mean_elevation']],
                    on='LANDKREIS',
                    how='left')
landkreisgdf_cl_new.info()

# %%
#Drop geometry and create a regular DataFrame
landkreisdf_cl = landkreisgdf_cl_new.drop(columns='geometry').copy()

#Save as parquet
save_path = os.path.join(
    data_main_path, "interim", "landkreisdf", "landkreisdf_crop_elev.parquet"
)
if not os.path.exists(save_path):
    landkreisdf_cl.to_parquet(save_path, index=False)
    print(f"Saved landkreisdf_cl → {save_path}")
else:
    print(f"landkreisdf_cl already exists, skipping save → {save_path}")


