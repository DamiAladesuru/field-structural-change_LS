# %%
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "workspace": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

os.makedirs("reports/figures/farm_field", exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

from src.data.processing_districtdata_utils import landkreis_desc as ld

# %%
farmcountdf= pd.read_parquet(data_main_path+"/interim/landkreisdf/farmcount_change.parquet")
_, gdf = ld.to_gdf(farmcountdf)

# %%
fig, ax = plt.subplots(figsize=(10, 12))

cmap = plt.get_cmap('Greys')
vmin = gdf['farm_count'].min()
vmax = gdf['farm_count'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

gdf.plot(
    column='farm_count',
    cmap=cmap,
    norm=norm,
    linewidth=0.3,
    edgecolor='white',
    legend=True,
    legend_kwds={'label': "Number of farms (2016)", 'shrink': 0.6},
    ax=ax
)

# Overlay distinct border styling for kreisfrei districts
kreisfrei_mask = gdf['LANDKREIS'].str.contains('kreisfrei', case=False, na=False)
gdf[kreisfrei_mask].boundary.plot(ax=ax, edgecolor='purple', linewidth=1, linestyle='--')

def get_text_color(value, norm, cmap):
    """Pick black or white text based on the luminance of this district's fill color."""
    rgba = cmap(norm(value))
    r, g, b = rgba[0], rgba[1], rgba[2]
    # Standard relative luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.5 else 'white'

for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    pct = row['farm_count_percdiff_to_y2016']
    if pd.notna(pct) and pd.notna(row['farm_count']):
        label = f"{pct:+.1f}%"
        text_color = get_text_color(row['farm_count'], norm, cmap)
        halo_color = 'white' if text_color == 'black' else 'black'

        ax.annotate(label, xy=(centroid.x, centroid.y), ha='center', fontsize=7,
                    color=text_color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1, foreground=halo_color)])

#ax.set_title("Number of Farms per District (2016), with % Change in 2020")
ax.axis('off')

plt.tight_layout()
plt.savefig('reports/figures/revision/farmcount_2016_change2020.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(12, 12))

cmap = plt.get_cmap('Greys')
vmin = gdf['farm_count'].min()
vmax = gdf['farm_count'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

gdf.plot(
    column='farm_count',
    cmap=cmap,
    norm=norm,
    linewidth=0.3,
    edgecolor='white',
    legend=False,
    ax=ax
)

# Overlay distinct border styling for kreisfrei districts
kreisfrei_mask = gdf['LANDKREIS'].str.contains('kreisfrei', case=False, na=False)
gdf[kreisfrei_mask].boundary.plot(ax=ax, edgecolor='purple', linewidth=1, linestyle='--')

def get_text_color(value, norm, cmap):
    """Pick black or white text based on the luminance of this district's fill color."""
    rgba = cmap(norm(value))
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.5 else 'white'

# Threshold for "too long" -- adjust based on how crowded the map looks
NAME_LENGTH_THRESHOLD = 12

long_name_refs = []  # will hold (number, full_name) pairs for the side legend
ref_counter = 1

for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    name = row['LANDKREIS']
    if pd.isna(name):
        continue

    text_color = get_text_color(row['farm_count'], norm, cmap)
    halo_color = 'white' if text_color == 'black' else 'black'

    if len(name) > NAME_LENGTH_THRESHOLD:
        # Use a numbered reference on the map instead of the full name
        label = str(ref_counter)
        long_name_refs.append((ref_counter, name))
        ref_counter += 1
    else:
        label = name

    ax.annotate(label, xy=(centroid.x, centroid.y), ha='center', va='center', fontsize=8,
                color=text_color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1, foreground=halo_color)])

ax.axis('off')

# Build the side legend text for long names
if long_name_refs:
    legend_text = "\n".join([f"{num}: {name}" for num, name in long_name_refs])
    fig.text(1.02, 0.5, legend_text, transform=ax.transAxes,
              fontsize=10, va='center', ha='left')

plt.tight_layout()
plt.savefig('reports/figures/revision/district_names_map.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
