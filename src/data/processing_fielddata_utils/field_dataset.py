# %%
import pandas as pd
import geopandas as gpd
import os, gc

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


from src.data.processing_fielddata_utils import dataload as dl
from src.data.processing_fielddata_utils import eca_new as eca

''' This script contains functions for:
    - modifying gld to contain kulturcode descriptions.
    - extracting and saving field geometries separately
    - saving and loading gld without geometries
    - loading field data deometries for given years
'''

def process_nogeom_data(years=range(2012, 2025), min_area=0.0001, return_mode='without_small'):
    """
    Optimized GLD processing pipeline.

    Steps:
    1. Load kulturcode mapping (unique only).
    2. Get GLD paths for available years.
    3. For each year:
        - Skip if processed no-geometry file already exists.
        - Load GLD, merge kulturcode mapping.
        - Split into small-area rows (area_km2 < min_area) and remaining rows.
        - Save geometry parquet (row_id, geometry).
        - Save no-geometry parquet (full data minus small-area rows).
        - Save small-area rows parquet separately.
    4. Return dictionary of no-geometry DataFrames for processed years.

    Parameters:
    - return_mode: 'without_small' -> returns dict of data with small-area rows removed (per year)
                   'both'          -> returns dict of full data (small-area rows included, i.e. main + small concatenated) per year

    Optimizations:
    - Skip already-processed years (crash recovery).
    - Drop large variables after each iteration + gc.collect().
    - Minimal memory footprint (not storing all geoms in memory).
    """

    assert return_mode in ('without_small', 'both'), "return_mode must be 'without_small' or 'both'"

    # Setup directories
    gldgeoms_dir = os.path.join(data_main_path, "interim", "gld_geoms")
    gld_no_geoms_dir = os.path.join(data_main_path, "interim", "gld_no_geoms")
    small_area_dir = os.path.join(data_main_path, "interim", "small_area_rows")
    os.makedirs(gldgeoms_dir, exist_ok=True)
    os.makedirs(gld_no_geoms_dir, exist_ok=True)
    os.makedirs(small_area_dir, exist_ok=True)

    # Detect already existing no-geom files
    existing_years = {
        int(f.split("_")[-1].replace(".parquet", ""))
        for f in os.listdir(gld_no_geoms_dir)
        if f.startswith("gld_no_geom_") and f.endswith(".parquet")
    }
    if existing_years:
        print(f"Found cached no-geometry files for years: {sorted(existing_years)}")

    # Load kulturcode mapping once
    kulturcode_mastermap = eca.process_kulturcode(data_main_path, load_existing=True)
    kulturcode_map_unique = kulturcode_mastermap.drop_duplicates(subset=['old_kulturcode'])
    del kulturcode_mastermap
    gc.collect()

    # Load gld paths once
    gld_paths = dl.load_data(loadExistingData=True)

    # Output dict
    gld_out = {}

    for year in years:
        nogeom_path = os.path.join(gld_no_geoms_dir, f"gld_no_geom_{year}.parquet")
        small_path = os.path.join(small_area_dir, f"small_area_{year}.parquet")

        if year in existing_years:
            print(f"Loading {year} from file.")
            df_main = pd.read_parquet(nogeom_path)
            if return_mode == 'without_small':
                gld_out[year] = df_main
            else:
                if os.path.exists(small_path):
                    df_small = pd.read_parquet(small_path)
                    gld_out[year] = pd.concat([df_main, df_small], ignore_index=True)
                else:
                    print(f"Warning: small-area file missing for {year}, returning main only.")
                    gld_out[year] = df_main
            continue

        print(f"Processing {year} ...")
        gld_path = gld_paths[year]
        gld = gpd.read_parquet(gld_path)

        # Merge kulturcode
        gld = gld.rename(columns={'kulturcode': 'old_kulturcode'})
        gld = gld.merge(kulturcode_map_unique, on='old_kulturcode', how='left')
        gld = gld.drop(columns=['old_kulturcode', 'int_kulturcode'])
        gld = gld.rename(columns={'new_kulturcode': 'kulturcode'})

        # Save geometry (row_id + geometry only) — before filtering, keeps all geometries
        geom_path = os.path.join(gldgeoms_dir, f"geometry_{year}.parquet")
        if not os.path.exists(geom_path):
            gld[['row_id', 'geometry']].to_parquet(geom_path, index=False)
            print(f"Saved geometry for {year} → {geom_path}")
        else:
            print(f"Geometry for {year} already exists, skipping → {geom_path}")

        # Drop geometry column for tabular processing
        df_no_geom = gld.drop(columns=['geometry']).copy()

        # Split into small-area and main (without small-area)
        is_small = df_no_geom['area_km2'] < min_area
        df_small = df_no_geom[is_small].copy()
        df_main = df_no_geom[~is_small].copy()

        # Save main (without small-area rows)
        df_main.to_parquet(nogeom_path, index=False)
        print(f"Saved no-geometry data (area >= {min_area}) for {year} → {nogeom_path}")

        # Save small-area rows separately
        df_small.to_parquet(small_path, index=False)
        print(f"{year}: {len(df_small)} rows with area_km2 < {min_area} → saved to {small_path}")

        # Store in output dict
        if return_mode == 'without_small':
            gld_out[year] = df_main
        else:
            gld_out[year] = pd.concat([df_main, df_small], ignore_index=True)

        # Free memory
        del gld, df_no_geom, df_main, df_small
        gc.collect()

    return gld_out


# Helper to load geometry parquet for a year
def load_gldgeom_for_year(year: int, return_mode='both'):
    """
    return_mode: 'without_small' -> excludes row_ids found in the small-area file
                 'both'          -> returns all geometries (no filtering)
    """

    assert return_mode in ('without_small', 'both'), "return_mode must be 'without_small' or 'both'"

    geom_dir = os.path.join(data_main_path, "interim", "gld_geoms")
    geom_path = os.path.join(geom_dir, f"geometry_{year}.parquet")
    gdf = gpd.read_parquet(geom_path)

    if return_mode == 'both':
        return gdf

    small_path = os.path.join(data_main_path, "interim", "small_area_rows", f"small_area_{year}.parquet")
    if os.path.exists(small_path):
        df_small = pd.read_parquet(small_path)
        gdf = gdf[~gdf['row_id'].isin(df_small['row_id'])]
    else:
        print(f"Warning: small-area file not found for {year}, returning all geometries.")

    return gdf


# Helper to load just the small-area (outlier) rows for a given year
def load_small_area_for_year(year: int):
    """
    Loads the small-area (excluded outlier) rows saved by process_nogeom_data
    for a given year. Returns an empty DataFrame with a warning if the file
    doesn't exist (e.g., year not yet processed).
    """
    small_path = os.path.join(data_main_path, "interim", "small_area_rows", f"small_area_{year}.parquet")
    if not os.path.exists(small_path):
        print(f"Warning: small-area file not found for {year}.")
        return pd.DataFrame()
    return pd.read_parquet(small_path)