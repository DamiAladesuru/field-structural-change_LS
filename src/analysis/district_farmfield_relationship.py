# %% Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colors, cm
import statsmodels.formula.api as smf
from scipy import stats


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

from src.data.processing_griddata_utils import griddf_desc as gd

# %% load full data for field and farm change
field_farm = pd.read_parquet(data_main_path+"/interim/landkreisdf/field_farmchange.parquet")
'''field_farm is prepared in src/data/grid_farmchange.py'''
#_, field_farm = ld.to_gdf(field_farm)

# %%
#######################################################
# Figure 8
#######################################################
# Drop all other columns
df = field_farm.copy()

change_cols = [
    # farm change vars
    'mean_UAA_percdiff_to_y2016',
    'farm_count_percdiff_to_y2016',
    'total_UAA_percdiff_to_y2016',
    
    # field change vars
    'medareakm2_percdiff_to_y2016',
    'fields_percdiff_to_y2016',
    'areakm2_sum_percdiff_to_y2016',
    'medshape_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016'
]

# Create the label mapping
label_mapping = {
    'mean_UAA_percdiff_to_y2016': 'Δ Mean UAA (%)',
    'farm_count_percdiff_to_y2016': 'Δ Total Number of Farms (%)',
    'total_UAA_percdiff_to_y2016': 'Δ Total UAA (%)',

    'medareakm2_percdiff_to_y2016': 'Δ MedFS (%)',
    'areakm2_sum_percdiff_to_y2016': 'Δ Total Field Area (%)',
    'fields_percdiff_to_y2016': 'Δ Total Number of Fields (%)',
    'fields_perarea_percdiff_to_y2016': 'Δ FD (%)',
    'medshape_percdiff_to_y2016': 'Δ MedSHAPE (%)',
}

# %% facet grid
import matplotlib.colors as mcolors

# Define variable configs: (column, title, bins)
var_configs = [
    ("mean_UAA_percdiff_to_y2016", "Δ Mean UAA", [-20, -15, -10, -5, 0, 5, 10, 15, 20]),
    ("farm_count_percdiff_to_y2016", "Δ Total Number of Farms", [-20, -15, -10, -5, 0, 5, 10, 15, 20]),
    ("total_UAA_percdiff_to_y2016", "Δ Total UAA", [-20, -15, -10, -5, 0, 5, 10, 15, 20]),
    ("medareakm2_percdiff_to_y2016", "Δ MedFS", [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]),
    ("fields_percdiff_to_y2016", "Δ Total Number of Fields", [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]),
    ("areakm2_sum_percdiff_to_y2016", "Δ Total Field Area", [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]),
    ("medshape_percdiff_to_y2016", "Δ MedSHAPE", [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]),
    ("fields_perarea_percdiff_to_y2016", "Δ FD", [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]),
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
                filename="reports/figures/revision/choropleth_change_row1.svg")
 
# Bins for row 1 (confirmed shared across its panels — using the first entry)
row1_bins = row1_configs[0][2]
plot_standalone_legend(row1_bins,
                        filename="reports/figures/revision/legend_row1.svg")
 
# ============================================================
# Rows 2-3: maps + standalone legend
# ============================================================
n_rows23 = -(-len(row23_configs) // n_cols)  # ceiling division
plot_maps_only(row23_configs, n_cols=n_cols, n_rows=n_rows23,
                filename="reports/figures/revision/choropleth_change_row23.svg")
 
# Bins for rows 2-3 (confirmed shared across their panels — using the first entry)
row23_bins = row23_configs[0][2]
plot_standalone_legend(row23_bins,
                        filename="reports/figures/revision/legend_row23.svg")


# %%
############################################
# correlation
############################################
# landkreis and Grid level
df = field_farm.copy()
griddf = gd.silence_prints(gd.create_fullgriddf, return_mode='both', ref_year=2016)
griddf_cl, _ = gd.clean_griddf(griddf)

columns_to_keep = [
    'LANDKREIS', 'year',
    'medareakm2', 'fields_perarea', 'medshape',
    'medareakm2_percdiff_to_y2016',
    'areakm2_sum_percdiff_to_y2016',
    'fields_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016',
    'medshape_percdiff_to_y2016'
]

# Drop all other columns
griddf_cl = griddf_cl[columns_to_keep]

# Rename grid-level columns before merge to make the scale explicit
griddf_cl_renamed = griddf_cl.rename(columns={
    col: f"{col}_grid" for col in griddf_cl.columns
    if col not in ['LANDKREIS', 'CELLCODE', 'year']  # keep key/id columns as-is
})
griddf_cl_renamed = griddf_cl_renamed[griddf_cl_renamed['year'] == 2020] 
dfwith_grid = df.merge(griddf_cl_renamed, on='LANDKREIS')

exclude_list = [lk for lk in df['LANDKREIS'].unique() if '(kreisfrei)' in lk]
print(exclude_list)  # verify what's being excluded before filtering

df_cl = df[~df['LANDKREIS'].isin(exclude_list)]
dfwith_grid_cl = dfwith_grid[~dfwith_grid['LANDKREIS'].isin(exclude_list)]


# %%
###################################################################
# kde to see distributon of landkreis level change of all variables
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
#plt.savefig("reports/figures/revision/landkreis_change_kde.tiff", dpi=300, bbox_inches='tight')
plt.savefig("reports/figures/revision/landkreis_change_kde.svg", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Define specific pairs (x, y)
corr_pairs = [
    ('mean_UAA_percdiff_to_y2016', 'medareakm2_percdiff_to_y2016'),
    ('farm_count_percdiff_to_y2016', 'medareakm2_percdiff_to_y2016'),
    ('mean_UAA_percdiff_to_y2016', 'fields_perarea_percdiff_to_y2016'),
    ('farm_count_percdiff_to_y2016', 'fields_perarea_percdiff_to_y2016'),
    ('mean_UAA_percdiff_to_y2016', 'medshape_percdiff_to_y2016'),
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
    'mean_UAA_percdiff_to_y2016',    # col 1
    'farm_count_percdiff_to_y2016',       # col 2
]

n_rows = len(field_vars)
n_cols = len(farm_vars)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex='col', sharey='row')

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
plt.savefig('reports/figures/revision/farm_field_scatter_wtotl_nofit.tiff', dpi=300, bbox_inches='tight')
plt.savefig('reports/figures/revision/farm_field_scatter_wtoutl_nofit.jpeg', dpi=300, bbox_inches='tight')
plt.show()



# %%
