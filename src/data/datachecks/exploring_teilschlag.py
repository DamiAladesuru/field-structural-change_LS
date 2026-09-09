# %%
# ===============================
# Imports & Project Setup
# ===============================
import os, io
from pathlib import Path
import requests
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt


# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "workspace": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break

project_root=os.getcwd()

# set the path to the data folder
data_main_path=open(project_root+"/datapath.txt").read()

# %%
url = open(data_main_path+"/raw/schlaege_2012.txt").read()

# Download the zip into memory
response = requests.get(url)
response.raise_for_status()

# Extract to a local folder
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("schlaege_2012")

# %%
# Read the shapefile (adjust filename if needed)
gdf = gpd.read_file("schlaege_2012/ud_12_s.shp") 
gdf_ts = gpd.read_file("schlaege_2012/ud_12_ts.shp") 


# %%
print(len(gdf))
print(len(gdf_ts))
print(gdf.groupby(['FLIK', 'SCHLAG_NR']).ngroups)
kulturcode_counts = gdf.groupby(['FLIK', 'SCHLAG_NR'])['KULTURCODE'].nunique().reset_index(name='n_kulturcode')
multi_kulturcode = kulturcode_counts[kulturcode_counts['n_kulturcode'] > 1]
print(f"Groups with more than 1 unique KULTURCODE: {len(multi_kulturcode)}")

# %%
joined = gpd.overlay(gdf_ts, gdf, how='intersection', keep_geom_type=False)
joined['overlap_area'] = joined.geometry.area

# after overlay, the original gdf_ts index is usually lost — need to preserve it before overlay
gdf_ts = gdf_ts.copy()
gdf_ts['ts_area'] = gdf_ts.geometry.area
gdf_ts['ts_id'] = gdf_ts.index  # preserve original id

joined = gpd.overlay(gdf_ts, gdf, how='intersection', keep_geom_type=False)
joined['overlap_area'] = joined.geometry.area
joined['overlap_ratio'] = joined['overlap_area'] / joined['ts_area']

best_overlap = joined.groupby('ts_id')['overlap_ratio'].max()
mismatched_ids = best_overlap[best_overlap < 0.99].index
mismatched = gdf_ts.set_index('ts_id').loc[mismatched_ids]

print(f"Rows with <99% overlap with any gdf geometry: {len(mismatched)}")

# %%
subset = gdf[(gdf['FLIK'] == "DENILI0341310128")]
subset

# %%
subset1 = gdf[(gdf['FLIK'] == "DENILI0341310128")]
subset2 = gdf_ts[(gdf_ts['FLIK'] == "DENILI0341310128")]
subset2['TEILSCHLAG'] = subset2['SCHLAG_NR'].astype(str) + subset2['TSCHLAG_BE'].astype(str)

fig, ax = plt.subplots(figsize=(8, 8))

subset1.plot(ax=ax, edgecolor='black', facecolor='lightgreen', alpha=0.6, label='subset1')
subset2.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5, label='subset2')

for idx, row in subset1.iterrows():
    centroid = row.geometry.centroid
    label = f"FLIK: {row['FLIK']}\nSCHLAG_NR: {row['SCHLAG_NR']}"
    ax.annotate(label, xy=(centroid.x, centroid.y), ha='right', fontsize=8, color='black')

for idx, row in subset2.iterrows():
    centroid = row.geometry.centroid
    label = f"FLIK: {row['FLIK']}\nTSCHLAG: {row['TEILSCHLAG']}"
    ax.annotate(label, xy=(centroid.x, centroid.y), ha='left', fontsize=8, color='red')

ax.set_title(f"schlag subset overlaid with teilschlag on same parcel)")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()


# %% another random example
subset1 = gdf[(gdf['FLIK'] == "DENILI1165800002")]
subset2 = gdf_ts[(gdf_ts['FLIK'] == "DENILI1165800002")]
subset2['TEILSCHLAG'] = subset2['SCHLAG_NR'].astype(str) + subset2['TSCHLAG_BE'].astype(str)

fig, ax = plt.subplots(figsize=(8, 8))

subset1.plot(ax=ax, edgecolor='black', facecolor='lightgreen', alpha=0.6, label='subset1')
subset2.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5, label='subset2')

for idx, row in subset1.iterrows():
    centroid = row.geometry.centroid
    label = f"FLIK: {row['FLIK']}\nSCHLAG_NR: {row['SCHLAG_NR']}\nKULTURCODE: {row['KULTURCODE']}"
    ax.annotate(label, xy=(centroid.x, centroid.y), ha='right', fontsize=8, color='black')

for idx, row in subset2.iterrows():
    centroid = row.geometry.centroid
    label = f"FLIK: {row['FLIK']}\nTSCHLAG: {row['TEILSCHLAG']}"
    ax.annotate(label, xy=(centroid.x, centroid.y), ha='left', fontsize=8, color='red')

ax.set_title(f"schlag subset overlaid with teilschlag on same parcel)")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

# %%
