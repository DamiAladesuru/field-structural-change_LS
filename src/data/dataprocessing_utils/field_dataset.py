# %%
import pandas as pd
import geopandas as gpd
import os, gc
import math as m

from pathlib import Path

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "workspace": # lower_saxony_fisc or workspace
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

from src.data.dataprocessing_utils import dataload as dl
from src.data.dataprocessing_utils import eca_new as eca

''' This script contains functions for:
    - modifying gld to contain kulturcode descriptions.
    - extracting and saving field geometries separately
    - saving and loading gld without geometries
    - loading field data deometries for given years
'''

# Function for shape metric
def shapeindex(p, a):
    return (0.282*p)/(m.sqrt(a))


def process_nogeom_data(
    years=range(2012, 2025),
    min_area=0.0001,
    categories_to_drop=(
        'Sonstige Flächen',
        'Stilllegung/Aufforstung',
        'Aus der Produktion/Erzeugung genommen',
        'AUKM',
        'GLK',
        'Sonstige Ökoregeln',
    ),
    return_mode='without_small',
):
    """
    Optimized GLD processing pipeline.

    Steps per year:
    1. Load kulturcode mapping (unique only) and merge onto GLD (adds 'Gruppe').
    2. Derive area_km2, peri_km, shape from the m-based columns.
    3. Save full geometry (row_id, geometry) before any row filtering.
    4. Set aside non-productive rows (Gruppe in categories_to_drop) -> saved separately.
    5. From the remaining productive rows, set aside small-area rows (area_km2 < min_area)
       -> saved separately.
    6. What's left ("main") is productive AND area_km2 >= min_area -> saved as the
       no-geometry file.

    Each of the three row sets (main, small-area, non-productive) is cached to its own
    parquet file per year, so any of them can be recombined later without recomputation.

    Parameters:
    - return_mode: which cached pieces to recombine into the returned DataFrame per year.
        'without_small'  -> main only (excludes small-area AND non-productive rows). [default]
        'productive'     -> main + small-area rows (non-productive excluded).
        'all'            -> main + small-area + non-productive rows (full reconstruction
                             of the original per-year dataset).

    Optimizations:
    - Skip already-processed years (crash recovery), based on the main no-geometry files.
    - Drop large variables after each iteration + gc.collect().
    - Minimal memory footprint (not storing all geoms in memory).
    """

    valid_return_modes = ('without_small', 'productive', 'with_nonprod', 'all')
    assert return_mode in valid_return_modes, f"return_mode must be one of {valid_return_modes}"

    categories_to_drop = list(categories_to_drop)

    # Setup directories
    gldgeoms_dir = os.path.join(data_main_path, "interim", "gld_geoms")
    gld_no_geoms_dir = os.path.join(data_main_path, "interim", "gld_no_geoms")
    small_area_dir = os.path.join(data_main_path, "interim", "small_area_rows")
    nonprod_dir = os.path.join(data_main_path, "interim", "nonprod_rows")
    os.makedirs(gldgeoms_dir, exist_ok=True)
    os.makedirs(gld_no_geoms_dir, exist_ok=True)
    os.makedirs(small_area_dir, exist_ok=True)
    os.makedirs(nonprod_dir, exist_ok=True)

    # Detect already-processed years (based on the main no-geometry files)
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

    def _assemble(df_main, df_small, df_nonprod):
        """Combine the cached pieces according to return_mode."""
        if return_mode == 'without_small':
            return df_main
        if return_mode == 'productive':
            return pd.concat([df_main, df_small], ignore_index=True)
        # 'all'
        return pd.concat([df_main, df_small, df_nonprod], ignore_index=True)

    for year in years:
        nogeom_path = os.path.join(gld_no_geoms_dir, f"gld_no_geom_{year}.parquet")
        small_path = os.path.join(small_area_dir, f"small_area_{year}.parquet")
        nonprod_path = os.path.join(nonprod_dir, f"nonprod_{year}.parquet")

        if year in existing_years:
            print(f"Loading {year} from file.")
            df_main = pd.read_parquet(nogeom_path)

            df_small = pd.DataFrame()
            if return_mode in ('productive', 'all'):
                if os.path.exists(small_path):
                    df_small = pd.read_parquet(small_path)
                else:
                    print(f"Warning: small-area file missing for {year}, continuing without it.")

            df_nonprod = pd.DataFrame()
            if return_mode in ('with_nonprod', 'all'):
                if os.path.exists(nonprod_path):
                    df_nonprod = pd.read_parquet(nonprod_path)
                else:
                    print(f"Warning: non-productive file missing for {year}, continuing without it.")

            gld_out[year] = _assemble(df_main, df_small, df_nonprod)
            continue

        print(f"Processing {year} ...")
        gld_path = gld_paths[year]
        gld = gpd.read_parquet(gld_path)

        # Merge kulturcode (adds 'Gruppe' among other columns)
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

        # 1. Set aside non-productive rows (by Gruppe)
        is_nonprod = df_no_geom['Gruppe'].isin(categories_to_drop)
        df_nonprod = df_no_geom[is_nonprod].copy()
        df_productive = df_no_geom[~is_nonprod].copy()

        # 2. From the productive rows, set aside small-area rows
        is_small = df_productive['area_km2'] < min_area
        df_small = df_productive[is_small].copy()
        df_main = df_productive[~is_small].copy()

        # 3. Save each piece
        df_main.to_parquet(nogeom_path, index=False)
        print(f"Saved main data (productive, area_km2 >= {min_area}) for {year} → {nogeom_path}")
        print(f"{year}: {len(df_main)} productive rows with area_km2 > {min_area}")

        df_nonprod.to_parquet(nonprod_path, index=False)
        print(f"{year}: {len(df_nonprod)} rows in non-productive categories {categories_to_drop} → saved to {nonprod_path}")

        df_small.to_parquet(small_path, index=False)
        print(f"{year}: {len(df_small)} productive rows with area_km2 < {min_area} → saved to {small_path}")

        # Store in output dict
        gld_out[year] = _assemble(df_main, df_small, df_nonprod)

        # Free memory
        del gld, df_no_geom, df_productive, df_main, df_small, df_nonprod
        gc.collect()

    return gld_out


# Helper to load geometry parquet for a year
def load_gldgeom_for_year(year: int):
    geom_dir = os.path.join(data_main_path, "interim", "gld_geoms")
    geom_path = os.path.join(geom_dir, f"geometry_{year}.parquet")
    gdf = gpd.read_parquet(geom_path)

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