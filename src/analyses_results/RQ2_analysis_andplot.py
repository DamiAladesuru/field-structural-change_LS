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

    if parent.name == "workspace": # lower_saxony_fisc or workspace
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/figures/farm_field", exist_ok=True)

# load utility modules
from src.data.coarseaggregates_andchange_utils import landkreis_desc as ld

#####################################################################
# makea df of farm and field change per landkreis
#####################################################################
# Compute farm value differences using years 2020 and 2016
#%% Load the cleaned DataFrame from CSV
farmdata = pd.read_csv(data_main_path+'/interim/farmdata.csv')
farmdata_ext = farmdata.copy()

# drop 2010
farmdata_ext = farmdata_ext[farmdata_ext['year']!= 2010]
farmdata_ext.reset_index(drop=True, inplace=True)

#resort
farmdata_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
numeric_columns = farmdata_ext.select_dtypes(include='number').columns

# Create a dictionary to store the new columns
new_columns = {}

# Loop through each numeric column
for col in numeric_columns:
    # Difference from farm's first year value within each group
    new_columns[f'{col}_diff_to_y2016'] = (
        farmdata_ext[col] - farmdata_ext.groupby('LANDKREIS')[col].transform('first')
    )

    # Percentage difference within each group
    new_columns[f'{col}_percdiff_to_y2016'] = (
        (farmdata_ext[col] - farmdata_ext.groupby('LANDKREIS')[col].transform('first')) / 
        farmdata_ext.groupby('LANDKREIS')[col].transform('first')
    ).fillna(0) * 100


# Concatenate the new columns to the original DataFrame all at once
new_columns_df = pd.DataFrame(new_columns)
farmdata_ydiff = pd.concat([farmdata_ext, new_columns_df], axis=1)
farmdata_ydiff.head()

farmdata_ydiff = farmdata_ydiff[farmdata_ydiff['year']== 2020]

# Create df with field and farm data
# %% Final df: field_farm
# One row per LANDKREIS combining farm and field data for the 2016 -> 2020 change analysis.
# Columns:
#   - farm_count_2016, total_UAA_km2_2016   : baseline farm levels (farmdata, 2016)
#   - farm_count_2020, total_UAA_km2_2020   : farm levels in the target year (farmdata_ydiff, 2020)
#   - *_percdiff_to_y2016 (farm)        : % change in farm variables, 2016 -> 2020
#   - medareakm2_2020, fields_perarea_2020,
#     medshape_2020                     : field-level values in 2020 (landkreisdf_cl)
#   - *_percdiff_to_y2016 (field)       : % change in field variables, 2016 -> 2020
#   - main_crop, main_crop_group        : dominant crop per Landkreis (summed over all years)
# %%
# Landkreis level field change
landkreisdf = ld.create_fulllandkreisdf(return_mode='without_small', gld_no_geom=None, ref_year = 2016)
landkreisdf_cl = ld.clean_landkreisdf(landkreisdf)
_, landkreis_yearly_cl = ld.silence_prints(ld.desc_landkreis, landkreisdf_cl)


base_year, target_year = 2016, 2020
fieldcolumns_to_keep = [
    'LANDKREIS', 'year',
    'medareakm2', 'fields_perarea', 'medshape',
    'medareakm2_percdiff_to_y2016',
    'areakm2_sum_percdiff_to_y2016',
    'fields_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016',
    'medshape_percdiff_to_y2016'
]
# --- 2016 baseline levels from farmdata
farm_base = (farmdata.loc[farmdata['year'] == base_year, ['LANDKREIS', 'farm_count', 'total_UAA_km2', 'mean_UAA_km2']]
             .rename(columns={'farm_count': f'farm_count_{base_year}',
                              'total_UAA_km2': f'total_UAA_km2_{base_year}',
                              'mean_UAA_km2': f'mean_UAA_km2_{base_year}'}))

# --- 2020 farm levels + changes vs 2016
farm_target = (farmdata_ydiff[farmdata_ydiff['year'] == target_year]
               .drop(columns='year')
               .rename(columns={'farm_count': f'farm_count_{target_year}',
                                'total_UAA_km2': f'total_UAA_km2_{target_year}',
                                'mean_UAA_km2': f'mean_UAA_km2_{target_year}'}))

# --- 2020 field levels + changes vs 2016
level_cols = ['medareakm2', 'fields_perarea', 'medshape']
field_target = (landkreisdf_cl.loc[landkreisdf_cl['year'] == target_year, fieldcolumns_to_keep]
                .drop(columns='year')
                .rename(columns={c: f'{c}_{target_year}' for c in level_cols}))

# --- merge: one row per Landkreis
field_farm = (farm_base
              .merge(farm_target, on='LANDKREIS', how='outer', validate='1:1', indicator='_m1')
              .merge(field_target, on='LANDKREIS', how='outer', validate='1:1', indicator='_m2'))

# unmatched Landkreise (name mismatches, missing years) show up here
print(field_farm[(field_farm['_m1'] != 'both') | (field_farm['_m2'] != 'both')][['LANDKREIS', '_m1', '_m2']])
field_farm = field_farm.drop(columns=['_m1', '_m2'])

field_farm.info()
field_farm.to_parquet(data_main_path + "/interim/landkreisdf/field_farmchange.parquet")

# %%
#####################################################################
# plot choropleth of farm count in 2016 with % change to 2020
#####################################################################
farmcountdf = field_farm[['LANDKREIS', f'farm_count_{base_year}', f'total_UAA_km2_{base_year}', 'farm_count_percdiff_to_y2016']]
_, gdf = ld.to_gdf(farmcountdf)
fig, ax = plt.subplots(figsize=(10, 12))

cmap = plt.get_cmap('Greys')
vmin = gdf[f'farm_count_{base_year}'].min()
vmax = gdf[f'farm_count_{base_year}'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

gdf.plot(
    column=f'farm_count_{base_year}',
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
    if pd.notna(pct) and pd.notna(row[f'farm_count_{base_year}']):
        label = f"{pct:+.1f}%"
        text_color = get_text_color(row[f'farm_count_{base_year}'], norm, cmap)
        halo_color = 'white' if text_color == 'black' else 'black'

        ax.annotate(label, xy=(centroid.x, centroid.y), ha='center', fontsize=7,
                    color=text_color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1, foreground=halo_color)])

#ax.set_title("Number of Farms per District (2016), with % Change in 2020")
ax.axis('off')

plt.tight_layout()
plt.savefig('reports/figures/farmcount_2016_change2020.svg', dpi=300, bbox_inches='tight')
plt.show()

# %%
#####################################################################
# plot choropleth of district names
#####################################################################
fig, ax = plt.subplots(figsize=(12, 12))

cmap = plt.get_cmap('Greys')
vmin = gdf[f'farm_count_{base_year}'].min()
vmax = gdf[f'farm_count_{base_year}'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

gdf.plot(
    column=f'farm_count_{base_year}',
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

    text_color = get_text_color(row[f'farm_count_{base_year}'], norm, cmap)
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
plt.savefig('reports/figures/district_names_map.svg', dpi=300, bbox_inches='tight')
plt.show()

# %%
#######################################################
# Figure 7
#######################################################
#%%
_, df = ld.to_gdf(field_farm)
exclude_list = [lk for lk in df['LANDKREIS'].unique() if '(kreisfrei)' in lk]

mask_excl = df['LANDKREIS'].isin(exclude_list)
df_incl, df_excl = df[~mask_excl], df[mask_excl]

change_cols = [
    # farm change vars
    'mean_UAA_km2_percdiff_to_y2016',
    'farm_count_percdiff_to_y2016',
    'total_UAA_km2_percdiff_to_y2016',
    
    # field change vars
    'medareakm2_percdiff_to_y2016',
    'fields_percdiff_to_y2016',
    'areakm2_sum_percdiff_to_y2016',
    'medshape_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016'
]

# Create the label mapping
label_mapping = {
    'mean_UAA_km2_percdiff_to_y2016': 'Δ Mean UAA (%)',
    'farm_count_percdiff_to_y2016': 'Δ Total Number of Farms (%)',
    'total_UAA_km2_percdiff_to_y2016': 'Δ Total UAA (%)',

    'medareakm2_percdiff_to_y2016': 'Δ MedFS (%)',
    'areakm2_sum_percdiff_to_y2016': 'Δ Total Field Area (%)',
    'fields_percdiff_to_y2016': 'Δ Total Number of Fields (%)',
    'fields_perarea_percdiff_to_y2016': 'Δ FD (%)',
    'medshape_percdiff_to_y2016': 'Δ MedSHAPE (%)',
}

# %% facet grid
# Define variable configs: (column, title, bins)
var_configs = [
    ("mean_UAA_km2_percdiff_to_y2016", "Δ Mean UAA", [-20, -15, -10, -5, 0, 5, 10, 15, 20]),
    ("farm_count_percdiff_to_y2016", "Δ Total Number of Farms", [-20, -15, -10, -5, 0, 5, 10, 15, 20]),
    ("total_UAA_km2_percdiff_to_y2016", "Δ Total UAA", [-20, -15, -10, -5, 0, 5, 10, 15, 20]),
    ("medareakm2_percdiff_to_y2016", "Δ MedFS", [-7, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 7]),
    ("fields_percdiff_to_y2016", "Δ Total Number of Fields", [-7, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 7]),
    ("areakm2_sum_percdiff_to_y2016", "Δ Total Field Area", [-7, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 7]),
    ("medshape_percdiff_to_y2016", "Δ MedSHAPE", [-7, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 7]),
    ("fields_perarea_percdiff_to_y2016", "Δ FD", [-7, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 7]),
]

n_cols = 3
n_rows = -(-len(var_configs) // n_cols)  # ceiling division

cmap = plt.get_cmap('coolwarm')
 
# Split var_configs into the two groups
row1_configs = var_configs[0:n_cols]
row23_configs = var_configs[n_cols:]
 
# Data aspect ratio (used to size map panels so they aren't padded with white space)
minx, miny, maxx, maxy = df.total_bounds
data_aspect = (maxx - minx) / (maxy - miny)
panel_height = 5
 
 
def plot_maps_only(configs, n_cols, n_rows, filename):
    """Plots the maps only, no legend, tightly cropped to data bounds."""
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(data_aspect * panel_height * n_cols, panel_height * n_rows))
    axes = np.atleast_1d(axes).flatten()
 
    for i, (col, title, bins) in enumerate(configs):
        ax = axes[i]
        norm = mcolors.BoundaryNorm(bins, cmap.N)

        df_incl.plot(column=col, cmap=cmap, norm=norm, linewidth=0.2,
                    edgecolor='grey', ax=ax, legend=False)

        # excluded districts: neutral fill, same outline, drawn on top
        df_excl.plot(ax=ax, facecolor='#e6e6e6', edgecolor='grey', linewidth=0.2,
                    hatch='////', zorder=3)

        ax.set_title(title, fontsize=11)
        ax.axis('off')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
 
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
 
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.05, wspace=-0.1, hspace=0.05)
    fig.savefig(filename, format="svg", dpi=300, bbox_inches='tight')
    plt.show()
    return fig
 
 
def plot_standalone_legend(bins, filename, figsize=(6, 1)):
    """Generates a standalone horizontal colorbar with ticks at each bin edge."""
    norm = mcolors.BoundaryNorm(bins, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
 
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
 
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=1.0, pad=0)
 
    cbar.set_ticks(bins)
    cbar.set_ticklabels([f"{b:g}" for b in bins])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Percentage Change (%)", fontsize=10)
 
    fig.savefig(filename, format="svg", dpi=300, bbox_inches='tight')
    plt.show()
    return fig
 
 
# ============================================================
# Row 1: maps + standalone legend
# ============================================================
plot_maps_only(row1_configs, n_cols=n_cols, n_rows=1,
                filename="reports/figures/choropleth_change_row1_masked.svg")
 
# Bins for row 1 (confirmed shared across its panels — using the first entry)
row1_bins = row1_configs[0][2]
plot_standalone_legend(row1_bins,
                        filename="reports/figures/legend_row1.svg")
 
# ============================================================
# Rows 2-3: maps + standalone legend
# ============================================================
n_rows23 = -(-len(row23_configs) // n_cols)  # ceiling division
plot_maps_only(row23_configs, n_cols=n_cols, n_rows=n_rows23,
                filename="reports/figures/choropleth_change_row23_masked.svg")
 
# Bins for rows 2-3 (confirmed shared across their panels — using the first entry)
row23_bins = row23_configs[0][2]
plot_standalone_legend(row23_bins,
                        filename="reports/figures/legend_row23.svg")


# for all 45 regions
# Drop all other columns
_, df = ld.to_gdf(field_farm)

n_cols = 3
n_rows = -(-len(var_configs) // n_cols)  # ceiling division

cmap = plt.get_cmap('coolwarm')
 
# Split var_configs into the two groups
row1_configs = var_configs[0:n_cols]
row23_configs = var_configs[n_cols:]
 
# Data aspect ratio (used to size map panels so they aren't padded with white space)
minx, miny, maxx, maxy = df.total_bounds
data_aspect = (maxx - minx) / (maxy - miny)
panel_height = 5
 
 
def plot_maps_only(configs, n_cols, n_rows, filename):
    """Plots the maps only, no legend, tightly cropped to data bounds."""
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(data_aspect * panel_height * n_cols, panel_height * n_rows))
    axes = np.atleast_1d(axes).flatten()
 
    for i, (col, title, bins) in enumerate(configs):
        ax = axes[i]
        norm = mcolors.BoundaryNorm(bins, cmap.N)
 
        df.plot(column=col, cmap=cmap, norm=norm, linewidth=0.2,
                edgecolor='grey', ax=ax, legend=False)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
 
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
 
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.05, wspace=-0.1, hspace=0.05)
    fig.savefig(filename, format="svg", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

 
# ============================================================
# Row 1: maps + standalone legend
# ============================================================
plot_maps_only(row1_configs, n_cols=n_cols, n_rows=1,
                filename="reports/figures/choropleth_change_row1.svg")
 
# ============================================================
# Rows 2-3: maps + standalone legend
# ============================================================
n_rows23 = -(-len(row23_configs) // n_cols)  # ceiling division
plot_maps_only(row23_configs, n_cols=n_cols, n_rows=n_rows23,
                filename="reports/figures/choropleth_change_row23.svg")
 

# %%
df = field_farm.copy()
exclude_list = [lk for lk in df['LANDKREIS'].unique() if '(kreisfrei)' in lk]
print(exclude_list)  # verify what's being excluded before filtering
# create df copy without kreisfrei
df_cl = df[~df['LANDKREIS'].isin(exclude_list)]
kreisfrei_df = df[df['LANDKREIS'].isin(exclude_list)]

cols = [
    'farm_count_percdiff_to_y2016', 'total_UAA_km2_percdiff_to_y2016', 'mean_UAA_km2_percdiff_to_y2016',
    'medareakm2_percdiff_to_y2016', 'areakm2_sum_percdiff_to_y2016', 'fields_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016', 'medshape_percdiff_to_y2016',
]

summary = pd.DataFrame({
    col: {
        'positive': (df_cl[col] > 0).sum(),
        'negative': (df_cl[col] < 0).sum(),
        'zero': (df_cl[col] == 0).sum(),
        'missing': df_cl[col].isna().sum(),
    } for col in cols
}).T

summary['total'] = summary.sum(axis=1)
print(summary)

# %%
cols = [
    'farm_count_percdiff_to_y2016', 'total_UAA_km2_percdiff_to_y2016', 'mean_UAA_km2_percdiff_to_y2016',
    'medareakm2_percdiff_to_y2016', 'areakm2_sum_percdiff_to_y2016', 'fields_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016', 'medshape_percdiff_to_y2016',
]

summary = pd.DataFrame({
    col: {
        'positive': (kreisfrei_df[col] > 0).sum(),
        'negative': (kreisfrei_df[col] < 0).sum(),
        'zero': (kreisfrei_df[col] == 0).sum(),
        'missing': kreisfrei_df[col].isna().sum(),
    } for col in cols
}).T

summary['total'] = summary.sum(axis=1)
print(summary)

# %%
############################################
# correlation pplots for fig 8 and appendix
############################################
# landkreis and Grid level
from src.data.coarseaggregates_andchange_utils import griddf_desc as gd
griddf = gd.silence_prints(gd.create_fullgriddf, return_mode='without_small', ref_year=2016)
griddf_cl, _ = gd.clean_griddf(griddf)

# Drop all other columns
griddf_cl = griddf_cl[fieldcolumns_to_keep]

# Rename grid-level columns before merge to make the scale explicit
griddf_cl_renamed = griddf_cl.rename(columns={
    col: f"{col}_grid" for col in griddf_cl.columns
    if col not in ['LANDKREIS', 'CELLCODE', 'year']  # keep key/id columns as-is
})
griddf_cl_renamed = griddf_cl_renamed[griddf_cl_renamed['year'] == 2020] 
dfwith_grid = df.merge(griddf_cl_renamed, on='LANDKREIS')
dfwith_grid_cl = dfwith_grid[~dfwith_grid['LANDKREIS'].isin(exclude_list)]


# %%
cols = ['mean_UAA_km2_percdiff_to_y2016'] # 'farm_count_2016', 'farm_count_2020'

# kreisfrei only = rows in df that are not in df_cl
df_kf = df[~df['LANDKREIS'].isin(df_cl['LANDKREIS'])]

means = pd.DataFrame({
    'all (df)': df[cols].mean(),
    'without kreisfrei (df_cl)': df_cl[cols].mean(),
    'kreisfrei only': df_kf[cols].mean(),
}).T
means['n_districts'] = [len(df), len(df_cl), len(df_kf)]

print(means.round(1))


# %%
###################################################################
# kde to see distributon of landkreis level change of all variables
# using df with all districts
###################################################################
# Get all unique columns from pairs
unique_cols = change_cols

n_cols = 3
n_rows = -(-len(unique_cols) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(unique_cols):
    sns.kdeplot(df[col].dropna(), ax=axes[i], fill=True)
    label = label_mapping.get(col, col)
    axes[i].set_title("")
    axes[i].set_xlabel(label)

# Turn off unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.25)
plt.savefig("reports/figures/landkreis_change_kde.svg", dpi=300, bbox_inches='tight')
plt.show()

# %%
import scipy.stats as stats
# Define specific pairs (x, y)
corr_pairs = [
    ('mean_UAA_km2_percdiff_to_y2016', 'medareakm2_percdiff_to_y2016'),
    ('farm_count_percdiff_to_y2016', 'medareakm2_percdiff_to_y2016'),
    ('mean_UAA_km2_percdiff_to_y2016', 'fields_perarea_percdiff_to_y2016'),
    ('farm_count_percdiff_to_y2016', 'fields_perarea_percdiff_to_y2016'),
    ('mean_UAA_km2_percdiff_to_y2016', 'medshape_percdiff_to_y2016'),
    ('farm_count_percdiff_to_y2016', 'medshape_percdiff_to_y2016'),
]
results = []

for x_col, y_col in corr_pairs:
    row = {'x': x_col, 'y': y_col}

    for label, data in [('with_outlier', df), ('without_outlier', df_cl)]:
        subset = data[[x_col, y_col]].dropna()

        pearson_r, pearson_p = stats.pearsonr(subset[x_col], subset[y_col])
        spearman_r, spearman_p = stats.spearmanr(subset[x_col], subset[y_col])

        row[f'pearson_r_{label}'] = pearson_r
        row[f'pearson_p_{label}'] = pearson_p
        row[f'spearman_r_{label}'] = spearman_r
        row[f'spearman_p_{label}'] = spearman_p
        row[f'n_{label}'] = len(subset)

    results.append(row)

results_df = pd.DataFrame(results)
# Save to CSV
results_df.to_csv("reports/statistics/correlations.csv", index=False)
print(results_df)

# %%
# Define grid structure
field_vars = [
    'medareakm2_percdiff_to_y2016',       # row 1: MedFS
    'fields_perarea_percdiff_to_y2016',   # row 2: Fields per total area
    'medshape_percdiff_to_y2016',         # row 3: MedSHAPE
]
farm_vars = [
    'mean_UAA_km2_percdiff_to_y2016',    # col 1
    'farm_count_percdiff_to_y2016',       # col 2
]

n_rows = len(field_vars)
n_cols = len(farm_vars)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex='col', sharey='row')

# plot correlation with all districts
for row, field_col in enumerate(field_vars):
    grid_field_col = f"{field_col}_grid"

    for col, farm_col in enumerate(farm_vars):
        ax = axes[row, col]

        # Grid-level points (blue)
        grid_data = dfwith_grid[[farm_col, grid_field_col]].dropna()
        ax.scatter(grid_data[farm_col], grid_data[grid_field_col],
                   color='#004488', alpha=0.3, s=10) # or  #000080

        # Landkreis-level points (red)
        lk_data = df[[farm_col, field_col]].dropna()
        ax.scatter(lk_data[farm_col], lk_data[field_col],
                   color="#FF0022", alpha=0.8, s=5)

        # Fit line using Landkreis-level data only
        pearson_r, pearson_p = stats.pearsonr(lk_data[farm_col], lk_data[field_col])
        ax.text(0.05, 0.95, f"r = {pearson_r:.2f}",
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Dotted blue reference line at y = 0
        ax.axhline(0, color='#004488', linestyle=':', linewidth=1)

        # Axis labels only on outer edges
        if col == 0:
            ax.set_ylabel(label_mapping.get(field_col, field_col), fontsize=10)
        else:
            ax.set_ylabel("")



        if row == n_rows - 1:
            ax.set_xlabel(label_mapping.get(farm_col, farm_col), fontsize=10)
        else:
            ax.set_xlabel("")

        # Legend on top-left plot only
        if row == 0 and col == 0:
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#004488',
                           alpha=0.5, markersize=6, label='Grid points'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0022',
                           alpha=0.8, markersize=6, label='Landkreis points'),
                plt.Line2D([0], [0], color='#004488', linestyle=':', linewidth=1, label='No FiSC (y=0)'),
            ]
            ax.legend(handles=legend_handles, fontsize=7, loc='best', frameon=True)
 

plt.tight_layout()
plt.savefig('reports/figures/farm_field_scatter_wtotl_nofit.tiff', dpi=300, bbox_inches='tight')
plt.savefig('reports/figures/farm_field_scatter_wtoutl_nofit.jpeg', dpi=300, bbox_inches='tight')
plt.show()



# %% plot correlation excluding kreisfree
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex='col', sharey='row')

for row, field_col in enumerate(field_vars):
    grid_field_col = f"{field_col}_grid"

    for col, farm_col in enumerate(farm_vars):
        ax = axes[row, col]

        # Grid-level points (blue)
        grid_data = dfwith_grid_cl[[farm_col, grid_field_col]].dropna()
        ax.scatter(grid_data[farm_col], grid_data[grid_field_col],
                   color='#004488', alpha=0.3, s=10) # or  #000080

        # Landkreis-level points (red)
        lk_data = df_cl[[farm_col, field_col]].dropna()
        ax.scatter(lk_data[farm_col], lk_data[field_col],
                   color="#FF0022", alpha=0.8, s=5)

        # Fit line using Landkreis-level data only
        pearson_r, pearson_p = stats.pearsonr(lk_data[farm_col], lk_data[field_col])
        ax.text(0.05, 0.95, f"r = {pearson_r:.2f}",
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Dotted blue reference line at y = 0
        ax.axhline(0, color='#004488', linestyle=':', linewidth=1)

        # Axis labels only on outer edges
        if col == 0:
            ax.set_ylabel(label_mapping.get(field_col, field_col), fontsize=10)
        else:
            ax.set_ylabel("")



        if row == n_rows - 1:
            ax.set_xlabel(label_mapping.get(farm_col, farm_col), fontsize=10)
        else:
            ax.set_xlabel("")

        # Legend on top-left plot only
        if row == 0 and col == 0:
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#004488',
                           alpha=0.5, markersize=6, label='Grid points'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0022',
                           alpha=0.8, markersize=6, label='Landkreis points'),
                plt.Line2D([0], [0], color='#004488', linestyle=':', linewidth=1, label='No FiSC (y=0)'),
            ]
            ax.legend(handles=legend_handles, fontsize=7, loc='best', frameon=True)
 

plt.tight_layout()
plt.savefig('reports/figures/farm_field_scatter_nofit.tiff', dpi=300, bbox_inches='tight')
plt.savefig('reports/figures/farm_field_scatter_nofit.jpeg', dpi=300, bbox_inches='tight')
plt.show()
