# %%
import geopandas as gpd
import pandas as pd
import os
import pickle
from pathlib import Path

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "lower_saxony_fisc": # or workspace if not lower_saxony_fisc 
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

# this script joins the eea grid with the landkreise boundaries of Niedersachsen
# and saves the result as a pickle file in the interim data folder.
# the result is a GeoDataFrame with the grid cells that intersect with the landkreise boundaries.
# the geodataframe is called and used in the dataload script to ssign landkreis and grid cell
# to the field data points.

def join_gridregion(loadExistingData=False):
    """
    Join 10km EEA grid cells with Niedersachsen Landkreise boundaries.
    Handles cells crossing multiple Landkreise by largest overlap,
    and fixes missing small Landkreise by reassigning if they cover ≥ 25% of a cell.
    """

    # Define default paths
    base_dir = data_main_path + "/raw"
    admin_files_dir = os.path.join(base_dir, "verwaltungseinheiten")
    intpath = data_main_path + "/interim"

    # Load precomputed data if available
    if loadExistingData and os.path.isfile(os.path.join(intpath, "grid_landkreise.pkl")):
        with open(os.path.join(intpath, "grid_landkreise.pkl"), "rb") as f:
            grid_landkreise = pickle.load(f)
        return grid_landkreise

    # --- Step 1: Load Landkreise boundaries ---
    landkreise = gpd.read_file(os.path.join(admin_files_dir, "NDS_Landkreise.shp"))
    landkreise = landkreise.to_crs("EPSG:25832")

    # --- Step 2: Load 10km grid ---
    grid = gpd.read_file(
        os.path.join(base_dir, "eea_10_km_eea-ref-grid-de_p_2013_v02_r00/de_10km.shp")
    )
    grid = grid.to_crs(landkreise.crs)
    grid_ = grid.reset_index().rename(columns={"index": "id"})

    # --- Step 3: Spatial join grid and landkreise ---
    grid_landkreise = gpd.sjoin(grid_, landkreise, how="inner", predicate="intersects")

    # --- Step 3b: Bounding box sanity check ---
    land = gpd.read_file(os.path.join(admin_files_dir, "NDS_Landesflaeche.shp"))
    land = land.to_crs("EPSG:25832")

    land_total_bounds = land.total_bounds
    grid_landkreise_total_bounds = grid_landkreise.total_bounds
    print("Land total bounds:", land_total_bounds)
    print("Grid-Landkreise total bounds:", grid_landkreise_total_bounds)

    if (land_total_bounds[0] >= grid_landkreise_total_bounds[0] and
        land_total_bounds[1] >= grid_landkreise_total_bounds[1] and
        land_total_bounds[2] <= grid_landkreise_total_bounds[2] and
        land_total_bounds[3] <= grid_landkreise_total_bounds[3]):
        print("All of land is within grid_landkr.")
    else:
        print("Some parts of land are not within grid_landkr.")

    # --- Step 4: Handle duplicates by largest overlap ---
    double_assigned = grid_landkreise[
        grid_landkreise.index.isin(grid_landkreise.index[grid_landkreise.index.duplicated()])
    ]

    # Remove duplicates first
    grid_landkreise = grid_landkreise[
        ~grid_landkreise.index.isin(grid_landkreise.index[grid_landkreise.index.duplicated()])
    ]

    # Compute intersection area (ha)
    double_assigned["intersection"] = [
        a.intersection(landkreise.loc[[b], "geometry"].values[0]).area / 10000
        for a, b in zip(double_assigned.geometry.values, double_assigned.index_right)
    ]

    # Keep only the row with largest intersection per cell
    double_assignedsorted = (
        double_assigned.sort_values(by="intersection")
        .groupby("id")
        .last()
        .reset_index()
    )

    # Add back into grid_landkreise
    grid_landkreise = pd.concat([grid_landkreise, double_assignedsorted])

    # --- Step 5: Fix missing landkreise (≥25% rule) ---
    all_lk = set(landkreise["LANDKREIS"].unique())
    assigned_lk = set(grid_landkreise["LANDKREIS"].unique())
    missing_lk = all_lk - assigned_lk
    print(f"Missing LANDKREISE after largest-overlap rule: {missing_lk}")

    if missing_lk:
        candidates = double_assigned[double_assigned["LANDKREIS"].isin(missing_lk)]
        fixed_rows = []

        for idx, row in candidates.iterrows():
            cell = row.geometry
            cell_area = cell.area

            # geometry of the missing landkreis
            lk_geom = landkreise.loc[landkreise.index == row["index_right"], "geometry"].values[0]
            overlap = cell.intersection(lk_geom).area
            proportion = overlap / cell_area

            if proportion >= 0.25:
                # keep row but add overlap_fraction info
                fixed_row = row.copy()
                fixed_row["overlap_fraction"] = proportion
                fixed_rows.append(fixed_row)

                # remove any conflicting assignment for this cell
                grid_landkreise = grid_landkreise[grid_landkreise["CELLCODE"] != row["CELLCODE"]]

        if fixed_rows:
            fixed_gdf = gpd.GeoDataFrame(fixed_rows, geometry="geometry", crs=landkreise.crs)
            grid_landkreise = pd.concat([grid_landkreise, fixed_gdf])
            grid_landkreise = grid_landkreise.set_crs(landkreise.crs, allow_override=True)

    # --- Step 6: Clean columns and save ---
    grid_landkreise = grid_landkreise.drop(
        columns=["id", "EOFORIGIN", "NOFORIGIN", "index_right", "LK", "intersection", "overlap_fraction"],
        errors="ignore",
    )

    # Save
    grid_landkreise.to_pickle(os.path.join(intpath, "grid_landkreise.pkl"))

    return grid_landkreise

# %%
grid_landkreise = join_gridregion(loadExistingData=True)