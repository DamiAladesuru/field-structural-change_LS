# %%
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from matplotlib import font_manager
print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "workspace": # or lower_saxony_fisc if not workspace
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

from src.data.processing_griddata_utils import griddf_desc as gd
os.makedirs("reports/figures", exist_ok=True)

'''Create the 2012, 2024 and difference plots together'''
# %% combined baseline and difference plots
def plot_choropleths_maps(gdf, year_list, columns, col_titles, bins_labels_dict, basecmap="Paired", diffcmap="coolwarm", suptitle=""):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    nrows = 3
    ncols = 5

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 16))  # Added extra height for colorbar

    # First column: Baseline values for 2012
    gdf_2012 = gdf[gdf['year'] == 2012]
    axes[0, 0].set_title("2012", fontsize=12)

    for row, column in enumerate(columns[:3]):
        bins, labels = bins_labels_dict[column]
        
        binned_column = f"{column}_bins"
        gdf_2012[binned_column] = pd.cut(gdf_2012[column], bins=bins, labels=labels, right=False)
        
        gdf_2012.plot(
            column=binned_column,
            cmap=basecmap,
            legend=True,
            legend_kwds={
                "loc": "lower left",
                "fontsize": "small",
                "title": None,
                #"title_fontsize": "x-small",
                "frameon": False,
                "labelspacing": 0.2,
                "alignment": "center",
                "borderpad": 2,
                "bbox_to_anchor": (0, -0.1),
            },
            edgecolor="white",
            linewidth=0.1,
            ax=axes[row, 0],
        )
        
        axes[row, 0].set_axis_off()
        axes[row, 0].text(-0.05, 0.5, col_titles[row], rotation=90, verticalalignment='center', 
                          horizontalalignment='center', transform=axes[row, 0].transAxes, fontsize=12)

    # Last column: Endline values for 2024
    gdf_2024 = gdf[gdf['year'] == 2024]
    axes[0, 4].set_title("2024", fontsize=12)

    for row, column in enumerate(columns[:3]):
        bins, labels = bins_labels_dict[column]
        
        binned_column = f"{column}_bins"
        gdf_2024[binned_column] = pd.cut(gdf_2024[column], bins=bins, labels=labels, right=False)
        
        gdf_2024.plot(
            column=binned_column,
            cmap=basecmap,
            legend=True,
            legend_kwds={
                "loc": "lower left",
                "fontsize": "small",
                "title": None,
                #"title_fontsize": "x-small",
                "frameon": False,
                "labelspacing": 0.2,
                "alignment": "center",
                "borderpad": 2,
                "bbox_to_anchor": (0, -0.1),
            },
            edgecolor="white",
            linewidth=0.1,
            ax=axes[row, 4],
        )
        
        axes[row, 4].set_axis_off()

    # Remaining columns: Percentage differences
    bins, labels = bins_labels_dict[columns[3]]  # Use the same bins and labels for all percentage difference columns
    norm = colors.BoundaryNorm(bins, plt.get_cmap(diffcmap).N)

    for col, year in enumerate(year_list, start=1):
        gdf_year = gdf[gdf['year'] == year]
        
        for row, column in enumerate(columns[3:]):
            binned_column = f"{column}_bins"
            gdf_year[binned_column] = pd.cut(gdf_year[column], bins=bins, labels=labels, right=False)
            
            gdf_year.plot(
                column=column,  # Use the actual values, not the binned ones
                cmap=diffcmap,
                norm=norm,
                legend=False,  # We'll add a shared legend later
                edgecolor="white",
                linewidth=0.1,
                ax=axes[row, col],
            )
            
            if row == 0:
                axes[row, col].set_title(f"{year}", fontsize=14)
            
            axes[row, col].set_axis_off()

    # Add a shared colorbar legend below the subplots
    scalar_mappable = plt.cm.ScalarMappable(cmap=diffcmap, norm=norm)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.3,
        aspect=43,
        shrink=1,
    )

    # Compute bin midpoints for tick placement
    bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    # Set tick positions and labels
    cbar.set_ticks(bin_midpoints)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=14)   # or whatever size you want

    cbar.set_label("Percentage Change from Base Year (2012)", fontsize=18)

    # Adjust layout using subplots_adjust
    fig.subplots_adjust(
        left=0.15,  # Adjust left margin
        right=0.9,  # Adjust right margin
        top=0.9,  # Adjust top margin
        bottom=0.15,  # Adjust bottom margin for the legend
        wspace=0,  # Adjust width between subplots
        hspace=0,  # Adjust height between subplots
    )
    
    # Add a super title
    fig.suptitle(suptitle, fontsize=22, y=0.93) # reduce y to move title nearer to plots
    
    #save plot as svg
    #plt.savefig("reports/figures/FiSC_choropleth_mapBerlin_new.svg", format="svg", bbox_inches='tight')
    #plt.savefig("reports/figures/revision/FiSC_choropleth_map.tiff", format="tiff", dpi=300, bbox_inches='tight')
    plt.savefig("reports/figures/revision/FiSC_choropleth_map.png", format="png", dpi=300, bbox_inches='tight')
    plt.savefig("reports/figures/revision/FiSC_choropleth_map.svg", format="svg", dpi=300, bbox_inches='tight')
    
    plt.show()
# %%
# Usage
columns = ["medareakm2", "fields_perarea", "medshape", 
           "medareakm2_percdiff_to_y1", "fields_perarea_percdiff_to_y1", "medshape_percdiff_to_y1"]
year_list = [2016, 2020, 2024]
col_titles = ["MedFS", "FD", "MedSHAPE"]

binsA = [-60, -6, -4, -2, -1, 0, 1, 2, 4, 6, 60]
labelsA = ["<-6%", "-5%", "-3%", "-1.5%", "-0.5%", "0.5%", "1.5%", "3%", "5%", ">6%"]

mfsbins = [0.5/100, 1.5/100, 2.0/100, 2.5/100, 3.5/100, 5.0/100]  # area variable: ha → km², divide by 100
mfslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(mfsbins[:-1], mfsbins[1:])]

fieldsbins = [0*100, 0.29*100, 0.33*100, 0.37*100, 0.43*100, 1.0*100]  # density: fields/ha → fields/km², multiply by 100
fieldslabels = [f"{low:.0f} - {high:.0f}" for low, high in zip(fieldsbins[:-1], fieldsbins[1:])]

shapebins = [1, 1.3, 1.35, 1.4, 1.5, 2.0]
shapelabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(shapebins[:-1], shapebins[1:])]

bins_labels_dict = {
    "medareakm2": (mfsbins, mfslabels),
    "fields_perarea": (fieldsbins, fieldslabels),
    "medshape": (shapebins, shapelabels),
    "medareakm2_percdiff_to_y1": (binsA, labelsA),
    "fields_perarea_percdiff_to_y1": (binsA, labelsA),
    "medshape_percdiff_to_y1": (binsA, labelsA)
}
# %% load data
griddf = gd.silence_prints(gd.create_fullgriddf, return_mode='without_small', ref_year=2016)
griddf_cl, _ = gd.clean_griddf(griddf)

# %% make gridgdf_cl
_, gridgdf_cl = gd.to_gdf(griddf_cl)
# print crs of gridgdf_cl
gridgdf_cl.crs
# re-project data
geoData = gridgdf_cl.to_crs(epsg=4326)

# %%
plot_choropleths_maps(geoData, year_list, columns, col_titles, bins_labels_dict, basecmap="Blues", diffcmap="coolwarm")
# suptitle="Field Structural Change Over Time"

# %%
################################################
#grid metric descriptive statics used in section 3.1
columns = ["medareakm2", "fields_perarea", "medshape"]
year_col = "year"

bins_labels_dict = {
    "medareakm2": (mfsbins, mfslabels),
    "fields_perarea": (fieldsbins, fieldslabels),
    "medshape": (shapebins, shapelabels),
}

results = []

for col in columns:
    bins, labels = bins_labels_dict[col]
    # cut values into bins
    griddf_cl[f"{col}_bin"] = pd.cut(
        griddf_cl[col], bins=bins, labels=labels, include_lowest=True
    )
    # count rows per (year, bin)
    counts = griddf_cl.groupby([year_col, f"{col}_bin"]).size().reset_index(name="count")
    # total counts per year
    totals = counts.groupby(year_col)["count"].transform("sum")
    # proportion
    counts["proportion"] = counts["count"] / totals
    # add metric column
    counts["metric"] = col
    

    # keep clean result
    results.append(counts)
    
# Combine all metrics into one long-format DataFrame
final_df = pd.concat(results, ignore_index=True)

# Save to CSV
final_df.to_csv("reports/statistics/map_metric_bin_proportions_new.csv",
                index=False, encoding="utf-8-sig")

# %%
################################################
##grid metric descriptive statistics describing maps in figure 6
columns = ["medareakm2_percdiff_to_y1", "fields_perarea_percdiff_to_y1", "medshape_percdiff_to_y1"]
year_col = "year"

binsA = [-60, -6, -4, -2, -1, 0, 1, 2, 4, 6, 60]
labelsA = ["<-6%", "-6 to -4%", "-4 to -2%", "-2 to -1%", "-1 to 0%", "0 to 1%", "1 to 2%", "2 to 4%", "4 to 6%", ">6%"]


bins_labels_dict = {
    "medareakm2_percdiff_to_y1": (binsA, labelsA),
    "fields_perarea_percdiff_to_y1": (binsA, labelsA),
    "medshape_percdiff_to_y1": (binsA, labelsA)
}

results = []

for col in columns:
    bins, labels = bins_labels_dict[col]
    # cut values into bins
    griddf_cl[f"{col}_bin"] = pd.cut(
        griddf_cl[col], bins=bins, labels=labels, include_lowest=True
    )
    # count rows per (year, bin)
    counts = griddf_cl.groupby([year_col, f"{col}_bin"]).size().reset_index(name="count")
    # total counts per year
    totals = counts.groupby(year_col)["count"].transform("sum")
    # proportion
    counts["proportion"] = counts["count"] / totals
    # add metric column
    counts["metric"] = col
    

    # keep clean result
    results.append(counts)
    
# Combine all metrics into one long-format DataFrame
final_df = pd.concat(results, ignore_index=True)

# Save to CSV
final_df.to_csv("reports/statistics/map_metricchange_bin_proportions_new.csv",
                index=False, encoding="utf-8-sig")
# %%
