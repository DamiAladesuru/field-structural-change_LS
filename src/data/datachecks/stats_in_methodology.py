# %%
# ===============================
# Imports & Project Setup
# ===============================
import os, gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity

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
# load utility functions
from src.data.dataprocessing_utils import field_dataset as fd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

# %%
#####################################################################
# Obtain main cultivated crop or kulturart FOR EACH landkreis for 2.1
#####################################################################
# Retained rows (productive, >= min_area)
years = range(2012, 2025)
allgld_no_geom = fd.process_nogeom_data(years=years, return_mode='all')

# landkreis geometries
base_dir = data_main_path + "/raw"
admin_files_dir = os.path.join(base_dir, "verwaltungseinheiten")
landkreise = gpd.read_file(os.path.join(admin_files_dir, "NDS_Landkreise.shp"))
landkreise = landkreise.to_crs("EPSG:25832")

# area_ha across years by LANDKREIS and kulturart
area_sum_dict = {}

for year, df in allgld_no_geom.items():
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

landkreis_crop = landkreise.merge(
    dominant_kulturart[['LANDKREIS', 'kulturart', 'Gruppe', 'total_area_ha']],
    on='LANDKREIS',
    how='left'
)
landkreis_crop.info()

# examine data in maps
def plot_landkreis_with_attribute(landkreis_gdf, attr_col, title):
    fig, ax = plt.subplots(figsize=(12, 12))
    landkreis_gdf.plot(column=attr_col, ax=ax, cmap='Dark2', legend=True, edgecolor='gray', linewidth=0.3)
    ax.set_title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
plot_landkreis_with_attribute(landkreis_crop, "Gruppe", "LANDKREIS - Dominant CropGroup")

## Based on the landkreis geometries (full copy from source) used, the map shows
# more coverage than is seen in the version in the paper because the version in the paper
# shows only the spatial extent studied having applied cleaning procedures and narrowing the
# study area to relevant parts.
# The copy in manuscript is produced after aggregating clean field data to landkreis level as follows:
# %%
from src.data.coarseaggregates_andchange_utils import landkreis_desc as ld
# Landkreis level
gld_no_geom = fd.process_nogeom_data(years=years, return_mode='without_small')
landkreisdf = ld.create_fulllandkreisdf(return_mode='without_small', gld_no_geom=gld_no_geom, ref_year = 2016)
landkreisdf_cl = ld.clean_landkreisdf(landkreisdf)
geom, landkreis_gdf = ld.to_gdf(landkreisdf_cl)

lanndkreis_maincrop = landkreis_gdf[['LANDKREIS', 'main_crop_group', 'geometry']]
# keep one row per LANDKREIS and drop year column
lanndkreis_maincrop = lanndkreis_maincrop.drop_duplicates(subset=['LANDKREIS']).reset_index(drop=True)
lanndkreis_maincrop = lanndkreis_maincrop.drop(columns=['year'], errors='ignore')
plot_landkreis_with_attribute(lanndkreis_maincrop, "main_crop_group", "LANDKREIS - Dominant CropGroup")
   
#%% we create a shapefile of the landkreis geometries with the dominant crop group attributeand
# export to ArcGIS for further styling.
# make directory
os.makedirs(os.path.join(data_main_path, "interim/landkreis_maincrop"), exist_ok=True)
lanndkreis_maincrop.to_file(os.path.join(data_main_path, "interim/landkreis_maincrop", "landkreis_maincrop.shp"))
# print crs
print(lanndkreis_maincrop.crs)


# %%
################################################################
# values used in 2.2 and Table A1
################################################################
# ===============================
# field data stats based on
# productive or non-prod groups
# ===============================
# helper for loading the dropped rows from cache
def load_pieces(folder, prefix, columns=None):
    path = os.path.join(data_main_path, "interim", folder)
    return pd.concat(
        [pd.read_parquet(os.path.join(path, f"{prefix}_{y}.parquet"), columns=columns).assign(year=y)
         for y in years],
        ignore_index=True,
    )

min_area=0.0001   

nonprod = load_pieces("nonprod_rows", "nonprod", ['row_id', 'Gruppe', 'area_km2'])
small = load_pieces("small_area_rows", "small_area", ['row_id', 'area_km2'])
main = pd.concat(gld_no_geom.values(), ignore_index=True)[['row_id', 'area_km2']]

# Totals (computed once)
n_main, area_main = len(main), main['area_km2'].sum()
n_small, area_small = len(small), small['area_km2'].sum()
n_nonprod, area_nonprod = len(nonprod), nonprod['area_km2'].sum()

n_all = n_main + n_small + n_nonprod
area_all = area_main + area_small + area_nonprod

print(f"Non-productive: {n_nonprod:,} polygons (avg {n_nonprod / len(years):,.0f}/year), "
      f"{area_nonprod:,.2f} km² (avg {area_nonprod / len(years):,.2f} km²/year)")

# Optional: non-productive per year
print(nonprod.groupby('year').agg(polygons=('row_id', 'count'), area_km2=('area_km2', 'sum')).round(2))

# Breakdown table: productive (retained + small-area) vs non-productive groups
by_group = (nonprod.groupby('Gruppe')
            .agg(polygons=('row_id', 'count'), area_km2=('area_km2', 'sum'))
            .sort_values('area_km2', ascending=False))

rows = [
    ('Productive', 'Total', n_main + n_small, area_main + area_small),
    ('Productive', 'Retained (main)', n_main, area_main),
    ('Productive', f'Small-area (< {min_area} km²)', n_small, area_small),
    ('Non-productive', 'Total', n_nonprod, area_nonprod),
    *[('Non-productive', g, int(r.polygons), r.area_km2) for g, r in by_group.iterrows()],
    ('All', 'Total', n_all, area_all),
]

summary = pd.DataFrame(rows, columns=['category', 'group', 'polygons', 'area_km2'])
summary['polygons_%'] = 100 * summary['polygons'] / n_all
summary['area_%'] = 100 * summary['area_km2'] / area_all
summary['mean_area_km2'] = summary['area_km2'] / summary['polygons']
summary['mean_area_ha'] = 100 * summary['mean_area_km2']  # km² -> ha
summary = summary.round({'polygons_%': 2, 'area_%': 2, 'area_km2': 2, 'mean_area_km2': 4, 'mean_area_ha': 2})

print(summary.to_markdown(index=False))

os.makedirs("reports/statistics", exist_ok=True)
summary.to_csv("reports/statistics/productive_vs_nonproductive_breakdown.csv", index=False)

# %% 
# ===============================
# presence/ absence of 100m² fields
# ===============================
def format_p(p):
    if p < 0.001:
        return "< 0.001"
    else:
        return f"{p:.3g}"

n_years = len(years)
n_cols = 4
n_rows = -(-n_years // n_cols)  # ceiling division

area_results = []
shape_results = []

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
axes = axes.flatten()

for i, year in enumerate(years):
    df_without_outliers = gld_no_geom[year]  # already loaded, no re-read
    df_small = load_pieces("small_area_rows", "small_area")

    # Sanity check: confirm no overlap between the two (should always be True given the pipeline)
    overlap = df_without_outliers['row_id'].isin(df_small['row_id']).sum()
    if overlap > 0:
        print(f"WARNING {year}: {overlap} row_id(s) unexpectedly found in both files.")

    # "With outliers" = the two combined
    df_with_outliers = pd.concat([df_without_outliers, df_small], ignore_index=True)

    # --- Area stats ---
    stats_with = df_with_outliers['area_km2'].describe()
    stats_without = df_without_outliers['area_km2'].describe()
    mw_stat_area, mw_p_area = stats.mannwhitneyu(df_with_outliers['area_km2'], df_without_outliers['area_km2'])

    area_results.append({
        'year': year,
        'n_all': len(df_with_outliers),
        'n_without_outliers': len(df_without_outliers),
        'mean_with': stats_with['mean'],
        'mean_without': stats_without['mean'],
        'median_with': stats_with['50%'],
        'median_without': stats_without['50%'],
        'std_with': stats_with['std'],
        'std_without': stats_without['std'],
        'mw_stat': mw_stat_area,
        'mw_p': mw_p_area,
    })

    # --- Shape stats ---
    shape_stats_with = df_with_outliers['shape'].describe()
    shape_stats_without = df_without_outliers['shape'].describe()
    mw_stat_shape, mw_p_shape = stats.mannwhitneyu(df_with_outliers['shape'], df_without_outliers['shape'])

    shape_results.append({
        'year': year,
        'n_all': len(df_with_outliers),
        'n_without_outliers': len(df_without_outliers),
        'mean_with': shape_stats_with['mean'],
        'mean_without': shape_stats_without['mean'],
        'median_with': shape_stats_with['50%'],
        'median_without': shape_stats_without['50%'],
        'std_with': shape_stats_with['std'],
        'std_without': shape_stats_without['std'],
        'mw_stat': mw_stat_shape,
        'mw_p': mw_p_shape,
    })

    # --- Scatterplot panel for this year ---
    ax = axes[i]
    ax.scatter(df_without_outliers['area_km2'], df_without_outliers['shape'],
               color='black', alpha=0.4, s=8, label='All')
    ax.scatter(df_small['area_km2'], df_small['shape'],
               color='blue', alpha=0.7, s=10, label='Outliers (area < 100 m²)')
    ax.set_title(str(year))
    ax.set_xlabel("Area (km²)")
    ax.set_ylabel("SHAPE")

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
os.makedirs(("reports/figures"), exist_ok=True)
plt.savefig("reports/figures/area_shape_outliers.jpeg", dpi=300, bbox_inches='tight')
plt.savefig("reports/figures/area_shape_outliers.tiff", dpi=300, bbox_inches='tight')
plt.show()

# --- Build appendix tables ---
os.makedirs(("reports/statistics"), exist_ok=True)
def build_appendix_table(results, unit_label=""):
    table = pd.DataFrame(results)
    area_cols = ['mean_with', 'mean_without', 'median_with', 'median_without', 'std_with', 'std_without']
    for col in area_cols:
        table[col] = table[col].round(3)
    table['mw_p'] = table['mw_p'].apply(format_p)

    table = table[[
        'year',
        'n_all', 'mean_with', 'median_with', 'std_with',
        'n_without_outliers', 'mean_without', 'median_without', 'std_without',
        'mw_p'
    ]]
    table.columns = [
        'Year',
        'N (With <100 m² fields)', 'Mean (With <100 m² fields)', 'Median (With <100 m² fields)', 'SD (With <100 m² fields)',
        'N (Without <100 m² fields)', 'Mean (Without <100 m² fields)', 'Median (Without <100 m² fields)', 'SD (Without <100 m² fields)',
        'Mann-Whitney p'
    ]
    return table

area_table = build_appendix_table(area_results, ", km²")
print("\n=== Area robustness table ===")
print(area_table.to_string(index=False))
area_table.to_csv("reports/statistics/robustness_check_table_area.csv", index=False)

shape_table = build_appendix_table(shape_results, "")
print("\n=== Shape robustness table ===")
print(shape_table.to_string(index=False))
shape_table.to_csv("reports/statistics/robustness_check_table_shape.csv", index=False)

# %%
# maximum shape without 100m² geoms
col = 'shape'
for year, df in sorted(gld_no_geom.items()):
    print(f"{year}: max {col} = {df[col].max()}")
    
# --- Print final total number of observations in each year
for year in sorted(gld_no_geom):
    print(f"{year}: Total observations = {len(gld_no_geom[year])}")

# --- Compute overall stats
total_observations = sum(len(v) for v in gld_no_geom.values())
num_years = len(gld_no_geom)
print(f"Grand total observations across all years = {total_observations}")
print(f"Mean observations per year = {total_observations / num_years:.2f}")

# --- Check total area of fields per year
for year in sorted(gld_no_geom):
    if 'area_km2' in gld_no_geom[year].columns:
        print(f"{year}: Total area = {gld_no_geom[year]['area_km2'].sum():,.2f}")

# --- Compute overall total and mean area
total_area = sum(df['area_km2'].sum() for df in gld_no_geom.values() if 'area_km2' in df.columns)
mean_area = total_area / num_years if num_years > 0 else 0

print(f"Total area per year = {total_area:,.2f} km²")
print(f"Mean area per year = {mean_area:,.2f} km²")

# --- Per-year field-level mean and max area
for year in sorted(gld_no_geom):
    if 'area_km2' in gld_no_geom[year].columns:
        a = gld_no_geom[year]['area_km2']
        print(f"{year}: Mean field area = {a.mean():.4f} km², Max field area = {a.max():.4f} km²")

# --- Overall (all years pooled)
all_area = pd.concat([df['area_km2'] for df in gld_no_geom.values() if 'area_km2' in df.columns])

print(f"\nMean field area (all polygons, all years) = {all_area.mean():.4f} km² ({all_area.mean() * 100:.2f} ha)")
print(f"Max field area (all years) = {all_area.max():.4f} km² ({all_area.max() * 100:.2f} ha)")

# ===============================
# Farm data descriptives
# ===============================
# Load the data
farmdata = pd.read_csv(data_main_path+'/interim/farmdata.csv')

# Display basic info about the dataset
print("Dat Info:")
print(f"Shape: {farmdata.shape}")
print(f"\nColumns: {farmdata.columns.tolist()}")
print(f"\nFirst few rows:")
print(farmdata.head())
print("\n" + "="*80 + "\n")

print(farmdata['year'].unique())
print(farmdata['LANDKREIS'].unique())

# Calculate yearly sums (across LANDKREIS for each year)
print("YEARLY SUMS (across all LANDKREIS for each year):")
print("-" * 50)

# Assuming there's a year column - if it has a different name, adjust accordingly
year_col = 'year' if 'year' in farmdata.columns else 'YEAR'

yearly_sum = farmdata.groupby(year_col).agg({
    'total_UAA_km2': 'sum',
    'farm_count': 'sum'
}).round(2)

print(yearly_sum)
print("\n" + "="*80 + "\n")

# Calculate and print means across years
yearly_sum_mean = yearly_sum['total_UAA_km2'].mean()
yearly_count_mean = yearly_sum['farm_count'].mean()
print(f"Mean of yearly total_UAA_km2 sums: {yearly_sum_mean:,.2f}")
print(f"Mean of yearly farm_count sums: {yearly_count_mean:,.2f}")
print("\n" + "="*80 + "\n")

# %%
# ===============================
# Grid-level descriptives
# ===============================
from src.data.coarseaggregates_andchange_utils import griddf_desc as gd
'''Here, we can see the grid level descriptives discussed in the last two paragraphs of section 2.2'''
griddf = gd.create_fullgriddf(return_mode='without_small', gld_no_geom=gld_no_geom, ref_year = 2016)
# I always want to load griddf and process clean griddf separately so I can have uncleaned data for comparison or sensitivity analysis
griddf_cl, outl = gd.clean_griddf(griddf)
_, grid_yearly_cl = gd.silence_prints(gd.desc_grid, griddf_cl)

# number of unique CELLCODES per year
unique_cellcodes_per_year = griddf.groupby("year")["CELLCODE"].nunique()
print(unique_cellcodes_per_year)

max_fields = griddf['fields'].max()
mean_fields = griddf['fields'].mean()
print(f"Max fields = {max_fields}")
print(f"Mean fields = {mean_fields:.2f}")

unique_cellcodes_per_year = griddf_cl.groupby("year")["CELLCODE"].nunique()
print(unique_cellcodes_per_year)

max_fields = griddf_cl['fields'].max()
mean_fields = griddf_cl['fields'].mean()
print(f"Mean fields = {mean_fields:.2f}")


# %%
# ===============================
# Shape descriptives for 2.3
# ===============================
# allgld_no_geom already loaded at the top of this script -- no need to reload

# --- 1. Shape filtering + plotting (field with shape = 1. row_id is J838091, year is 2021)
# Define target values and extract fitting rows
target_shape_values = [1.00]
filtered_gld_dict = {}

for year, df in allgld_no_geom.items():
    # Filter rows where 'shape' is approximately equal to any target value
    filtered = df[df["shape"].apply(lambda x: np.any(np.isclose(x, target_shape_values, atol=0.01)))]
    filtered_gld_dict[year] = filtered

# Concatenate all years into one DataFrame with a 'year' column
filtered_gld = pd.concat(
    [df.assign(year=year) for year, df in filtered_gld_dict.items()],
    ignore_index=True
)

# Display results
print(filtered_gld.head())
print(f"Final shape: {filtered_gld.shape}")
del filtered_gld_dict

# %%
# From filtered_gld keep only row where row_id = J838091
selected = filtered_gld[filtered_gld["row_id"] == "J838091"]

# 2021 geometries
geom2021 = fd.load_gldgeom_for_year(2021)
selected = selected.merge(
    geom2021[["row_id", "geometry"]],
    on="row_id",
    how="left"
)
selected = gpd.GeoDataFrame(selected, geometry="geometry", crs="EPSG:25832")

fig, ax = plt.subplots(figsize=(4, 4))
#fig.suptitle("Sample Fields with their SHAPE values", fontsize=14, y=0.9)

fixed_xlim = (-1.2, 1.2)
fixed_ylim = (-1.2, 1.2)

for idx, row in selected.iterrows():
    geom = row.geometry
    if geom is None:
        continue

    # Handle Polygon & MultiPolygon
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        continue

    for poly in polygons:
        # Normalize: center + scale
        centroid = poly.centroid
        centered = affinity.translate(poly, xoff=-centroid.x, yoff=-centroid.y)

        bounds = centered.bounds
        max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if max_dim == 0:
            continue
        scaled = affinity.scale(centered, xfact=2/max_dim, yfact=2/max_dim, origin=(0, 0))

        x, y = scaled.exterior.xy
        ax.plot(x, y, 'b-', linewidth=1)
        ax.fill(x, y, 'lightgray', alpha=0.5)

    # Add text with area and SHAPE
    ax.text(
        0, fixed_ylim[0] - 0.1,
        f"Area of field: {row['area_m2']:.0f} m²\nSHAPE: {row['shape']:.2f}",
        fontsize=8, ha='center', va='top'
    )

ax.set_xlim(fixed_xlim)
ax.set_ylim(fixed_ylim)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

plt.subplots_adjust(wspace=0.5)
os.makedirs("reports/figures/", exist_ok=True)
plt.savefig("reports/figures/circularfield.svg", format="svg", bbox_inches='tight')
plt.show()

# %%
# --- 2. Shape filtering + plotting (area ~0.02km², 2012)
target_area_values = [0.02]
filtered_gld_dict = {}

for year, df in allgld_no_geom.items():
    # Filter rows where 'area_km2' is approximately equal to any target value
    filtered = df[df["area_km2"].apply(lambda x: np.any(np.isclose(x, target_area_values, atol=0.01)))]
    filtered_gld_dict[year] = filtered

# Concatenate all years into one DataFrame with a 'year' column
filtered_gld = pd.concat(
    [df.assign(year=year) for year, df in filtered_gld_dict.items()],
    ignore_index=True
)

# Display results
print(filtered_gld.head())
print(f"Final shape: {filtered_gld.shape}")

del filtered_gld_dict

# %%
sample_2012 = filtered_gld[filtered_gld['row_id'].isin(['A1351', 'A483', 'A602'])]

# 2012 geometries
geom2012 = fd.load_gldgeom_for_year(2012)
sample_2012 = sample_2012.merge(
    geom2012[["row_id", "geometry"]],
    on="row_id",
    how="left"
)

sample_2012 = gpd.GeoDataFrame(sample_2012, geometry="geometry", crs="EPSG:25832")

# -----------------------------
fig, ax = plt.subplots(1, len(sample_2012), figsize=(4 * len(sample_2012), 4))
if len(sample_2012) == 1:
    ax = [ax]  # make iterable if only one axis

#fig.suptitle("Sample Fields with their SHAPE values", fontsize=14, y=0.9)

fixed_xlim = (-1.2, 1.2)
fixed_ylim = (-1.2, 1.2)

for i, (idx, row) in enumerate(sample_2012.iterrows()):
    geom = row.geometry
    if geom is None:
        continue

    # Handle Polygon & MultiPolygon
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        continue

    for poly in polygons:
        # Normalize: center + scale
        centroid = poly.centroid
        centered = affinity.translate(poly, xoff=-centroid.x, yoff=-centroid.y)

        bounds = centered.bounds  # (minx, miny, maxx, maxy)
        max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if max_dim == 0:
            continue
        scaled = affinity.scale(centered, xfact=2/max_dim, yfact=2/max_dim, origin=(0, 0))

        x, y = scaled.exterior.xy
        ax[i].plot(x, y, 'b-', linewidth=1)
        ax[i].fill(x, y, 'lightgray', alpha=0.5)

    ax[i].set_xlim(fixed_xlim)
    ax[i].set_ylim(fixed_ylim)
    ax[i].set_aspect('equal')
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    # Add text with area + shape
    ax[i].text(
        0, fixed_ylim[0] - 0.1,
        f"Area: {row['area_km2']:.2f} km²\nSHAPE: {row['shape']:.2f}",
        fontsize=8, ha='center', va='top'
    )

plt.subplots_adjust(wspace=0.5)
os.makedirs("reports/figures/", exist_ok=True)
plt.savefig("reports/figures/examiningshape2.svg", format="svg", bbox_inches='tight')
plt.show()
# %%
