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
# %%
# load utility functions
from src.data.processing_fielddata_utils import field_dataset as fd
from src.data.processing_griddata_utils import griddf_desc as gd
from src.data.processing_fielddata_utils import dataload as dl

# ensure interim data directory exists
os.makedirs(os.path.join(data_main_path, "interim"), exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

# %%
# ===============================
# Load processed field data ONCE, reused throughout this script
# ===============================
years = range(2012, 2025)
gld_no_geom = fd.process_nogeom_data(years=years, return_mode='without_small')

# %%
###################################################
# check effect of presence/ absence of 100m² fields
###################################################

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
    df_small = fd.load_small_area_for_year(year)

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
plt.savefig("reports/figures/area_shape_outliers.jpeg", dpi=300, bbox_inches='tight')
plt.savefig("reports/figures/area_shape_outliers.tiff", dpi=300, bbox_inches='tight')
plt.show()

# --- Build appendix tables ---
def build_appendix_table(results, unit_label):
    table = pd.DataFrame(results)
    area_cols = ['mean_with', 'mean_without', 'median_with', 'median_without', 'std_with', 'std_without']
    for col in area_cols:
        table[col] = table[col].round(3)
    table['mw_p'] = table['mw_p'].apply(format_p)

    table = table[[
        'year', 'n_all', 'n_without_outliers',
        'mean_with', 'mean_without',
        'median_with', 'median_without',
        'std_with', 'std_without',
        'mw_p'
    ]]
    table.columns = [
        'Year', 'N (with)', 'N (without)',
        f'Mean (with{unit_label})', f'Mean (without{unit_label})',
        f'Median (with{unit_label})', f'Median (without{unit_label})',
        f'SD (with{unit_label})', f'SD (without{unit_label})',
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
##############################################################
''' The next code lines allow to obtain the general field data descriptives
that were stated in section 2'''

# ===============================
# Load field data shapefiles, harmonize their columns,
# add shape, cellcode and region information
# ===============================
'''PS: This step takes about 140 minutes to complete on a standard laptop'''
gld_paths = dl.load_data(loadExistingData=True)


# %%
# ===============================
# Load Lower Saxony data without cellcode, region or shape metric, etc.
# just the Lower Saxony data
# with this, you can compare number of fields before the data was cleaned
# ===============================

gld_dir = data_main_path+"/interim/gld_per_year"

# Check paths exist for decision logic
init_files  = {year: f"{gld_dir}/gld_{year}.parquet" for year in years}
all_init_files_exist  = all(os.path.exists(path) for path in init_files.values())

if all_init_files_exist:
    gld_raw_paths = init_files

gld_raw_allyears = {}  # Will store year → GeodataFrame of data

for year in years:
    # Load the gld data for the current year
    gld_path = gld_raw_paths[year]
    gld = gpd.read_parquet(gld_path)
    print(f"{year}: CRS of data: EPSG:{gld.crs.to_epsg()}")

    gld_raw_allyears[year] = gld
del gld
gc.collect()

######################################################
# 2.2.1 Field data descriptives
# %%
# ===============================
# Field-level descriptive stats (raw data)
# ===============================
# --- Print total number of observations in each year
for year in sorted(gld_raw_allyears):
    print(f"{year}: Total observations = {len(gld_raw_allyears[year])}")

# --- Compute overall stats
total_observations = sum(len(v) for v in gld_raw_allyears.values())
num_years = len(gld_raw_allyears)
mean_observations = total_observations / num_years if num_years > 0 else 0

print(f"Grand total observations across all years = {total_observations}")
print(f"Mean observations per year = {mean_observations:.2f}")


# %%
# ===============================
# Field-level descriptive stats (processed data, no geometry)
# ===============================
# gld_no_geom already loaded at the top of this script -- no need to reload

# --- Print total number of observations in each year
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

print(f"Mean area per year = {mean_area:,.2f} ha")


# %%
# ===============================
# Plots (field-level)
# ===============================
'''
You can also visualize how the area changes over the years
'''
def plot_total_area(gld_no_geom):
    area_sums = {}
    for year in sorted(gld_no_geom):
        df = gld_no_geom[year]
        if 'area_km2' in df.columns:
            area_sums[year] = df['area_km2'].sum()
        else:
            print(f"{year}: No 'area' column found.")

    plt.figure(figsize=(10, 6))
    plt.plot(area_sums.keys(), area_sums.values(), marker='o')
    plt.title("Total Area by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Area")
    plt.grid(True)
    #plt.savefig("reports/figures/total_area_per_year.png", dpi=300)
    plt.show()

plot_total_area(gld_no_geom)

# %%
'''
Plot how total observation changes over the years
'''
# Collect counts
year_counts = {year: len(df) for year, df in gld_no_geom.items()}

# Sort by year
sorted_years = sorted(year_counts.keys())
counts = [year_counts[y] for y in sorted_years]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(sorted_years, counts, marker='o', linestyle='-', linewidth=2)
plt.title("Total Observations per Year", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Observations", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Add labels above points
for x, y in zip(sorted_years, counts):
    plt.text(x, y + max(counts)*0.01, str(y), ha="center", fontsize=9)

plt.tight_layout()
#plt.savefig("reports/figures/total_observations_per_year.png", dpi=300)
plt.show()

# %%
# ===============================
# Examine Yearly + overall stats (area)
# ===============================
# --- Yearly stats
for year in sorted(gld_no_geom):
    if 'area_km2' in gld_no_geom[year].columns:
        mean_val = gld_no_geom[year]['area_km2'].mean()
        min_val = gld_no_geom[year]['area_km2'].min()
        max_val = gld_no_geom[year]['area_km2'].max()
        print(f"{year}: mean = {mean_val:.2f}, min = {min_val:.2f}, max = {max_val:.2f}")

# --- Overall stats (all years combined)
all_data = pd.concat([df[['area_km2']] for df in gld_no_geom.values() if 'area_km2' in df.columns])

overall_mean = all_data['area_km2'].mean()
overall_min = all_data['area_km2'].min()
overall_max = all_data['area_km2'].max()

print("\n--- Overall across all years ---")
print(f"Mean = {overall_mean:.2f}, Min = {overall_min:.2f}, Max = {overall_max:.2f}")

####################################################################
# 2.2.2 Farm data descriptives
# Load the data
farm_anim = pd.read_csv(data_main_path+'/interim/farm_anim.csv')

# Display basic info about the dataset
print("Dataset Info:")
print(f"Shape: {farm_anim.shape}")
print(f"\nColumns: {farm_anim.columns.tolist()}")
print(f"\nFirst few rows:")
print(farm_anim.head())
print("\n" + "="*80 + "\n")

# %%
# Calculate yearly sums (across LANDKREIS for each year)
print("YEARLY SUMS (across all LANDKREIS for each year):")
print("-" * 50)

# Assuming there's a year column - if it has a different name, adjust accordingly
year_col = 'year' if 'year' in farm_anim.columns else 'YEAR'

yearly_sum = farm_anim.groupby(year_col).agg({
    'total_farm_area': 'sum',
    'farm_count': 'sum',
    'animalfarms': 'sum',
    'animal_count': 'sum'
}).round(2)

print(yearly_sum)
print("\n" + "="*80 + "\n")

# Calculate and print means across years
yearly_sum_mean = yearly_sum['total_farm_area'].mean()
yearly_count_mean = yearly_sum['farm_count'].mean()
yearly_animcount_mean = yearly_sum['animal_count'].mean()
yearly_animfarm_mean = yearly_sum['animalfarms'].mean()
print(f"Mean of yearly total_farm_area sums: {yearly_sum_mean:,.2f}")
print(f"Mean of yearly farm_count sums: {yearly_count_mean:,.2f}")
print(f"Mean of yearly animal_count sums: {yearly_animcount_mean:,.2f}")
print(f"Mean of yearly animalfarms sums: {yearly_animfarm_mean:,.2f}")
print("\n" + "="*80 + "\n")


####################################################################
# %% 2.2.3.
# ===============================
# Grid-level descriptives
# ===============================
'''Here, we can see the grid level descriptives also discussed in section 2.2'''
griddf = gd.silence_prints(gd.create_fullgriddf)
# I always want to load griddf and process clean griddf separately so I can have uncleaned data for comparison or sensitivity analysis
griddf_cl, _ = gd.clean_griddf(griddf)

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


###########################################################################
# %%
# ===============================
# Shape analysis (field-level)
# ===============================
''' Here you can examine shape as described in section 2.3.3'''
# gld_no_geom already loaded at the top of this script -- no need to reload

# --- 1. Shape filtering + plotting (field with shape = 1. row_id is J838091, year is 2021)
# Define target values and extract fitting rows
target_shape_values = [1.00]
filtered_gld_dict = {}

for year, df in gld_no_geom.items():
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

# From filtered_gld keep only row where row_id = J838091
selected = filtered_gld[filtered_gld["row_id"] == "J838091"]

# 2021 geometries
geom2021 = fd.load_gldgeom_for_year(2021, return_mode='both')
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

    # Add text with area, SHAPE, and cpar_25
    ax.text(
        0, fixed_ylim[0] - 0.1,
        f"Area of field: {row['area_km2']:.2f} km²\nSHAPE: {row['shape']:.2f}",
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
# --- 2. Shape filtering + plotting (area ~1.94, 2012)
target_area_values = [1.94]
filtered_gld_dict = {}

for year, df in gld_no_geom.items():
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
geom2012 = fd.load_gldgeom_for_year(2012, return_mode='both')
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
# --- Yearly stats for shape
for year in sorted(gld_no_geom):
    if 'shape' in gld_no_geom[year].columns:
        mean_val = gld_no_geom[year]['shape'].mean()
        min_val = gld_no_geom[year]['shape'].min()
        max_val = gld_no_geom[year]['shape'].max()
        med_val = gld_no_geom[year]['shape'].median()
        print(f"{year}: mean = {mean_val:.2f}, min = {min_val:.2f}, max = {max_val:.2f}, med = {med_val:.2f}")

# --- Overall stats (all years combined)
all_data = pd.concat([df[['shape']] for df in gld_no_geom.values() if 'shape' in df.columns])

overall_mean = all_data['shape'].mean()
overall_min = all_data['shape'].min()
overall_max = all_data['shape'].max()
overall_med = all_data['shape'].median()

print("\n--- Overall across all years ---")
print(f"Mean = {overall_mean:.2f}, Min = {overall_min:.2f}, Max = {overall_max:.2f}, Med = {overall_med:.2f}")

# %%
# --- Grid-level min/max shape
print(griddf_cl['medshape'].min())
print(griddf_cl['medshape'].max())


# %%
# ===============================
# 2.3.4 FiSC Illustration
# ===============================
# gld_no_geom already loaded at the top of this script -- reused here, no reload

# --- Print total number of observations in each year
for year in sorted(gld_no_geom):
    print(f"{year}: Total observations = {len(gld_no_geom[year])}")

# obtain 2020 sample
gld20 = gld_no_geom[2020]
sample = gld20[gld20['flik'] == 'DENILI1832290051']

# obtain 2023 sample
gld23 = gld_no_geom[2023]
sample2 = gld23[gld23['flik'] == 'DENILI1832290051']

# Calculate medians for sample
print("\nSAMPLE2020 Dataset:")
print("-" * 40)
median_area_sample = sample['area_km2'].median()
median_shape_sample = sample['shape'].median()

print(f"Median area_km2: {median_area_sample:,.2f}")
print(f"Median shape: {median_shape_sample:,.4f}")

# Calculate medians for sample2
print("\nSAMPLE2023 Dataset:")
print("-" * 40)
median_area_sample2 = sample2['area_km2'].median()
median_shape_sample2 = sample2['shape'].median()

print(f"Median area_km2: {median_area_sample2:,.2f}")
print(f"Median shape: {median_shape_sample2:,.4f}")

# Summary comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"\narea_km2 median difference: {abs(median_area_sample - median_area_sample2):,.2f}")
print(f"shape median difference: {abs(median_shape_sample - median_shape_sample2):,.4f}")


''' I think methodology descriptives end here'''

# ===============================
# 3.1. Max and min change values
# for the different metrics
# ===============================
# %%
# find the row with max medfs_ha_percdiff_to_y1
max_medfs_idx = griddf_cl["medfs_ha_percdiff_to_y1"].idxmax()

# find the row with min medfs_ha_percdiff_to_y1
min_medfs_idx = griddf_cl["medfs_ha_percdiff_to_y1"].idxmin()

# find the row with max fields_ha_percdiff_to_y1
max_fields_idx = griddf_cl["fields_ha_percdiff_to_y1"].idxmax()

# find the row with min fields_ha_percdiff_to_y1
min_fields_idx = griddf_cl["fields_ha_percdiff_to_y1"].idxmin()

# find the row with max medshape_percdiff_to_y1
max_medshape_idx = griddf_cl["medshape_percdiff_to_y1"].idxmax()

# find the row with min medshape_percdiff_to_y1
min_medshape_idx = griddf_cl["medshape_percdiff_to_y1"].idxmin()


print(griddf_cl.loc[min_medfs_idx, ["year", "medfs_ha_percdiff_to_y1", "CELLCODE", "LANDKREIS"]])
print(griddf_cl.loc[max_medfs_idx, ["year", "medfs_ha_percdiff_to_y1", "CELLCODE", "LANDKREIS"]])
print(griddf_cl.loc[min_fields_idx, ["year", "fields_ha_percdiff_to_y1", "CELLCODE", "LANDKREIS"]])
print(griddf_cl.loc[max_fields_idx, ["year", "fields_ha_percdiff_to_y1", "CELLCODE", "LANDKREIS"]])
print(griddf_cl.loc[min_medshape_idx, ["year", "medshape_percdiff_to_y1", "CELLCODE", "LANDKREIS"]])
print(griddf_cl.loc[max_medshape_idx, ["year", "medshape_percdiff_to_y1", "CELLCODE", "LANDKREIS"]])

# %%
# Group by 'year' and calculate useful stds across grids
grid_yearly_stds = griddf_cl.groupby('year').agg(
    med_fsha_percdiffy1_std=('medfs_ha_percdiff_to_y1', 'std'),
    medshape_percdiffy1_std=('medshape_percdiff_to_y1', 'std'),
    fields_ha_percdiffy1_std=('fields_ha_percdiff_to_y1', 'std')

).reset_index()

grid_yearly_stds = grid_yearly_stds[grid_yearly_stds['year'] != 2012]
grid_yearly_stds['med_fsha_percdiffy1_var'] = grid_yearly_stds['med_fsha_percdiffy1_std']**2
grid_yearly_stds['medshape_percdiffy1_var'] = grid_yearly_stds['medshape_percdiffy1_std']**2
grid_yearly_stds['fields_ha_percdiffy1_var'] = grid_yearly_stds['fields_ha_percdiffy1_std']**2

# Pooled Standard Deviation for each metric
av_size_std = (sum(grid_yearly_stds['med_fsha_percdiffy1_var'])/12)**0.5
print(av_size_std)

av_shape_std = (sum(grid_yearly_stds['medshape_percdiffy1_var'])/12)**0.5
print(av_shape_std)

av_fa_std = (sum(grid_yearly_stds['fields_ha_percdiffy1_var'])/12)**0.5
print(av_fa_std)


####################################################
# %% Additional checks (not used in paper)
# ===============================
# 2012 vs 2024 Comparisons (grid-level)
# ===============================
# medfs_ha, fields_ha, medshape stats comparisons
# %% medfs_ha
# For 2012
subset_2012 = griddf_cl[griddf_cl['year'] == 2012]
print("2012 - max:", subset_2012['medfs_ha'].max())
print("2012 - min:", subset_2012['medfs_ha'].min())

# For 2024
subset_2024 = griddf_cl[griddf_cl['year'] == 2024]
print("2024 - max:", subset_2024['medfs_ha'].max())
print("2024 - min:", subset_2024['medfs_ha'].min())

# %% fields_ha
# For 2012
subset_2012 = griddf_cl[griddf_cl['year'] == 2012]
print("2012 - max:", subset_2012['fields_ha'].max())
print("2012 - min:", subset_2012['fields_ha'].min())

# For 2024
subset_2024 = griddf_cl[griddf_cl['year'] == 2024]
print("2024 - max:", subset_2024['fields_ha'].max())
print("2024 - min:", subset_2024['fields_ha'].min())

# %% medshape
# For 2012
subset_2012 = griddf_cl[griddf_cl['year'] == 2012]
print("2012 - max:", subset_2012['medshape'].max())
print("2012 - min:", subset_2012['medshape'].min())

# For 2024
subset_2024 = griddf_cl[griddf_cl['year'] == 2024]
print("2024 - max:", subset_2024['medshape'].max())
print("2024 - min:", subset_2024['medshape'].min())