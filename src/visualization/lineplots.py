# %%
import os
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator

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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

from src.data.processing_griddata_utils import griddf_desc as gd
from src.data.processing_districtdata_utils import landkreis_desc as ld

# %%
# Grid level
griddf = gd.silence_prints(gd.create_fullgriddf, return_mode='without_small', ref_year=2016)
griddf_cl, outl = gd.clean_griddf(griddf)
_, grid_yearly_cl = gd.silence_prints(gd.desc_grid, griddf_cl, ref_year=2016)

# %%
# Landkreis level
landkreisdf = ld.silence_prints(ld.create_fulllandkreisdf, return_mode='without_small', ref_year=2016)
landkreisdf_cl = ld.clean_landkreisdf(landkreisdf)
_, landkreis_yearly_cl = ld.silence_prints(ld.desc_landkreis, landkreisdf_cl, ref_year=2016)

####################################################################
# 1. we can make a single line plot for each metric using yearly_cl to
# see overall aggregate trend of change
#####################################################################
# %%
def multiline_metrics(df, title, ylabel, metrics, save_path, color_mapping=None, format='png', dpi=300):
    """
    Function to plot multiple metrics over time with predefined colors and save the plot.

    Args:
        df (DataFrame): Dataframe containing the data to plot.
        title (str): Title of the plot.
        ylabel (str): Label for the Y-axis.
        metrics (dict): Dictionary mapping labels to column names.
        save_path (str): File path to save the plot.
        color_mapping (dict): Optional dictionary specifying colors for each metric label.
        format (str): Format to save the plot (e.g., 'png', 'pdf', 'svg').
        dpi (int): Resolution of the plot in dots per inch.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    # Use provided color mapping or default to seaborn's color cycle
    for label, column in metrics.items():
        color = color_mapping.get(label, None) if color_mapping else None
        sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)

    plt.title(title, fontsize=16, pad=12)
    plt.xlabel('Year', labelpad=12, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title='Metrics')

    # Save the plot
    plt.savefig(save_path, format=format, dpi=dpi)
    print(f"Plot saved to {save_path}")

    plt.show()

# %%
multiline_df = grid_yearly_cl
metrics = {
    'MedFS': 'medareakm2_percdiffy1_med',
    'MedSHAPE': 'medshape_percdiffy1_med',
    'FD': 'fields_perarea_percdiffy1_med'
}

color_mapping = {
    #https://personal.sron.nl/~pault/
    'MedFS': '#004488',
    'Median Perimeter': '#EE7733',
    'MedSHAPE': '#228833',
    'FD': '#CC3311',
}



multiline_metrics(
    df=multiline_df,
    title="Changes in Field Structure Metrics Across Years",
    ylabel="Aggregate Change (%) in Field Metric Value from 2012",
    metrics=metrics,
    format='svg',
    save_path="reports/figures/aggregate_line_trends_0826_with100m2.svg",
    color_mapping=color_mapping
)


################################################################################
# 2. Plot the figures in paper:
# Figure 4: grid disaggreated trend of change in field metric values over time
################################################################################
'''Goal is to create plot with grid trend lines and aggregate median line for each metric'''
# %% function for subplots of grid metrics value with aggregate median line
def create_fisc_trend_plots(fig, axs, griddf, grid_yearly, plot_configs, suptitle=None):
    """
    Create a 2x2 subplot of FiSC trend plots.
    
    Parameters:
    fig (Figure): Matplotlib Figure object
    axs (Array of Axes): 2x2 array of Matplotlib Axes objects
    griddf (DataFrame): DataFrame containing individual cell data
    grid_yearly (DataFrame): DataFrame containing aggregate yearly data
    plot_configs (list): List of dictionaries containing plot configurations
    suptitle (str): Super title for the entire figure
    """
    fig.suptitle(suptitle, fontsize=16)

    for config in plot_configs:
        ax = config['ax']
        
        # Remove border around plot
        [ax.spines[side].set_visible(False) for side in ax.spines]
        
        # Plot individual lines for each unique CELLCODE
        for cellcode in griddf['CELLCODE'].unique():
            data = griddf[griddf['CELLCODE'] == cellcode]
            ax.plot(data['year'], data[config['y_col']], color='grey', alpha=0.9, linewidth=0.5)
        
        # Plot the aggregate thick line and annotate
        line = ax.plot(grid_yearly['year'], grid_yearly[config['agg_col']], color='purple', linewidth=1.5)[0]
        
        # Annotate the end of the line
        last_year = grid_yearly['year'].iloc[-1]
        last_value = grid_yearly[config['agg_col']].iloc[-1]
        ax.annotate(f'{last_value:.2f}', 
                    xy=(last_year, last_value),
                    xytext=(5, 0),
                    textcoords='offset points',
                    color='black',
                    fontweight='bold',
                    ha='left',
                    va='center')
        
        ax.set_xlabel('Year', labelpad=12, fontsize=12, x=0.46)

        ax.set_title(config['title'])
        
        # Style the grid
        ax.grid(which='major', color='#EAEAF2', linewidth=1.2)
        ax.grid(which='minor', color='#EAEAF2', linewidth=0.6)
        ax.minorticks_on()
        ax.tick_params(which='minor', bottom=False, left=False)
        
        # Only show minor gridlines once in between major gridlines
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Adjust layout
    #plt.tight_layout()

# %%
fig, axs = plt.subplots(1, 3, figsize=(18, 5)) #for two rows, that would be plt.subplots(2, 2, figsize=(10, 10))
plot_configs = [
    {'ax': axs[0], 'title': 'MedFS', 'y_col': 'medareakm2_percdiff_to_y1', 'agg_col': 'medareakm2_percdiffy1_med'},
    {'ax': axs[1], 'title': 'FD', 'y_col': 'fields_perarea_percdiff_to_y1', 'agg_col': 'fields_perarea_percdiffy1_med'},
    {'ax': axs[2], 'title': 'MedSHAPE', 'y_col': 'medshape_percdiff_to_y1', 'agg_col': 'medshape_percdiffy1_med'}   
]
create_fisc_trend_plots(fig, axs, griddf_cl, grid_yearly_cl, plot_configs)

plt.subplots_adjust(left=0.09, wspace=0.2, hspace=0.2, top=0.80)
fig.text(0.05, 0.45, 'Percentage Change from Base Year (2012)', va='center', ha='center', rotation='vertical', fontdict={'fontsize': 12}) #, transform=fig.transFigure

#save plot as svg
plt.savefig("reports/figures/revision/FiSC_trendlines.jpeg", format="jpeg", bbox_inches='tight')

plt.show()

# %%
################################################################################
# Figure 5: Rotated Fisc kde with histogram for 2016, 2020, 2024
################################################################################
# Create the label mapping
label_mapping = {
    'medareakm2_percdiff_to_y1': 'Δ MedFS (%)',
    'fields_perarea_percdiff_to_y1': 'Δ FD (%)',
    'medshape_percdiff_to_y1': 'Δ MedSHAPE (%)',
}

# Variables and years
variables = ['medareakm2_percdiff_to_y1', 'fields_perarea_percdiff_to_y1', 'medshape_percdiff_to_y1']
years = [2016, 2020, 2024]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, var in enumerate(variables):
    for j, year in enumerate(years):
        ax = axes[i, j]
        
        # Filter data for year
        data = griddf_cl[griddf_cl['year'] == year][var].dropna().values
        
        # Histogram (density=True to match KDE scale)
        ax.hist(data, bins=30, density=True, alpha=0.5, color='steelblue', edgecolor='darkgrey',
                orientation="horizontal")
        
        # Gaussian KDE
        if len(data) > 1:
            kde = gaussian_kde(data)
            y_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(kde(y_range), y_range, color='black', linewidth=1)
        
        # Horivontal line at 0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        
        ax.set_ylim(data.min(), data.max())
        ax.set_xticks([])

        # Invert x-axis so peak is on left
        #ax.invert_xaxis()

        # Remove spines
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        

        # Years on top (column titles)
        if i == 0:
            ax.set_title(f'{year}', fontsize=12)
        
        # Variables on left (row labels)
        if j == 0:
            ax.set_ylabel(label_mapping.get(var, var), fontsize=12)
        else:
            ax.set_ylabel('')
        
        # Y-axis ticks on right side (last column only)
        if j == 0:
            ax.yaxis.tick_left()
        else:
            ax.set_yticklabels([])

plt.tight_layout()
plt.savefig('reports/figures/Fisckde_hist_3x3_rotatedwithoutlier.svg', dpi=300, bbox_inches='tight')
plt.show()


