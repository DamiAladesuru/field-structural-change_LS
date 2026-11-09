# %%
import pandas as pd
import geopandas as gpd
import os, gc
import logging
import numpy as np
import pickle
# Silence the print statements in a function call
import contextlib
import io
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

from src.data.dataprocessing_utils import field_dataset as fd

# %% B. Creating landkreisdf (LANDKREIS-level aggregation, parallel to griddf/CELLCODE version)
def create_landkreisdf_base(gld):
    """
    Create a landkreisdf DataFrame with aggregated statistics that summarize field data,
    aggregated by LANDKREIS instead of CELLCODE.

    Parameters:
    gld (DataFrame): Input DataFrame with columns ['year', 'LANDKREIS']
                        and additional numeric columns for aggregation.

    Returns:
    A DataFrame with unique LANDKREIS, year rows,
                  enriched with aggregated fields.
    """

    required_columns = ['year', 'LANDKREIS', 'row_id',
                         'Gruppe', 'area_km2', 'peri_km', 'shape']
    missing = [col for col in required_columns if col not in gld.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    # Step 1: Create base landkreisdf
    columns = ['year', 'LANDKREIS']
    landkreisdf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created landkreisdf with shape {landkreisdf.shape}")
    logging.info(f"Columns in landkreisdf: {landkreisdf.columns}")

    # Helper function for aggregation
    def add_aggregated_column(landkreisdf, gld, column, aggfunc, new_col):
        logging.info(f"Adding column '{new_col}' using '{aggfunc}' on '{column}'.")
        temp = gld.groupby(['LANDKREIS', 'year'])[column].agg(aggfunc).reset_index()
        temp.columns = ['LANDKREIS', 'year', new_col]
        return pd.merge(landkreisdf, temp, on=['LANDKREIS', 'year'], how='left')

    # Define aggregations
    aggregations = [
        {'column': 'row_id', 'aggfunc': 'count', 'new_col': 'fields'},
        {'column': 'Gruppe', 'aggfunc': 'nunique', 'new_col': 'group_count'},
        {'column': 'area_km2', 'aggfunc': 'sum', 'new_col': 'areakm2_sum'},
        {'column': 'peri_km', 'aggfunc': 'sum', 'new_col': 'peri_sum'},
        {'column': 'shape', 'aggfunc': 'sum', 'new_col': 'shape_sum'},
        {'column': 'area_km2', 'aggfunc': 'mean', 'new_col': 'mareakm2'},
        {'column': 'peri_km', 'aggfunc': 'mean', 'new_col': 'mperi'},
        {'column': 'shape', 'aggfunc': 'mean', 'new_col': 'mshape'},
        {'column': 'area_km2', 'aggfunc': 'median', 'new_col': 'medareakm2'},
        {'column': 'peri_km', 'aggfunc': 'median', 'new_col': 'medperi'},
        {'column': 'shape', 'aggfunc': 'median', 'new_col': 'medshape'},
    ]

    # Apply each aggregation
    for agg in aggregations:
        landkreisdf = add_aggregated_column(landkreisdf, gld, agg['column'], agg['aggfunc'], agg['new_col'])

    # Rate of fields per area of land per LANDKREIS
    landkreisdf['fields_perarea'] = (landkreisdf['fields'] / landkreisdf['areakm2_sum'])

    return landkreisdf


# check for duplicates in the landkreisdf
def check_duplicates_landkreis(landkreisdf):
    duplicates = landkreisdf[landkreisdf.duplicated(subset=['LANDKREIS', 'year'], keep=False)]
    print(f"Number of duplicates in landkreisdf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")


# yearly LANDKREIS differences and differences from first year
def calculate_yearlydiff_landkreis(landkreisdf):  # yearly LANDKREIS differences
    landkreisdf_ext = landkreisdf.copy()

    # Ensure the data is sorted by 'LANDKREIS' and 'year'
    landkreisdf_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
    numeric_columns = landkreisdf_ext.select_dtypes(include='number').columns

    new_columns = {}

    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = landkreisdf_ext.groupby('LANDKREIS')[col].diff().fillna(0)
        new_columns[f'{col}_yearly_percdiff'] = (
            landkreisdf_ext.groupby('LANDKREIS')[col].diff() /
            landkreisdf_ext.groupby('LANDKREIS')[col].shift(1)
        ).fillna(0) * 100

    new_columns_df = pd.DataFrame(new_columns)
    landkreisdf_ext = pd.concat([landkreisdf_ext, new_columns_df], axis=1)

    return landkreisdf_ext


def calculate_diff_fromy1_landkreis(landkreisdf):  # yearly differences from first year
    landkreisdf_ext = landkreisdf.copy()

    landkreisdf_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
    numeric_columns = landkreisdf_ext.select_dtypes(include='number').columns

    new_columns = {}

    # Get the first occurrence of each unique LANDKREIS
    y1_df = landkreisdf_ext.groupby('LANDKREIS').first().reset_index()

    y1_df = y1_df[['LANDKREIS'] + list(numeric_columns)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    landkreisdf_ext = pd.merge(landkreisdf_ext, y1_df, on='LANDKREIS', how='left')

    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = landkreisdf_ext[col] - landkreisdf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = (
            (landkreisdf_ext[col] - landkreisdf_ext[f'{col}_y1']) / landkreisdf_ext[f'{col}_y1']
        ) * 100

    landkreisdf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    new_columns_df = pd.DataFrame(new_columns)
    landkreisdf_exty1 = pd.concat([landkreisdf_ext, new_columns_df], axis=1)

    return landkreisdf_exty1

def calculate_diff_fromyear(df, group_col='LANDKREIS', ref_year=2016):
    """
    Calculate difference and percent difference from a specified reference year,
    for each group (CELLCODE or LANDKREIS).
    """
    df_ext = df.copy()
    df_ext.sort_values(by=[group_col, 'year'], inplace=True)
    numeric_columns = df_ext.select_dtypes(include='number').columns
    numeric_columns = [c for c in numeric_columns if c != 'year']

    if ref_year not in df_ext['year'].unique():
        raise ValueError(f"ref_year {ref_year} not found in df['year'] values.")

    ref_df = df_ext[df_ext['year'] == ref_year][[group_col] + list(numeric_columns)].copy()
    ref_df = ref_df.rename(columns={col: f'{col}_yref' for col in numeric_columns})

    df_ext = pd.merge(df_ext, ref_df, on=group_col, how='left')

    new_columns = {}
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y{ref_year}'] = df_ext[col] - df_ext[f'{col}_yref']
        new_columns[f'{col}_percdiff_to_y{ref_year}'] = (
            (df_ext[col] - df_ext[f'{col}_yref']) / df_ext[f'{col}_yref']
        ) * 100

    df_ext.drop(columns=[f'{col}_yref' for col in numeric_columns], inplace=True)

    new_columns_df = pd.DataFrame(new_columns)
    df_ext = pd.concat([df_ext, new_columns_df], axis=1)

    return df_ext


def combine_landkreisdfs(landkreisdf_ext, landkreisdf_exty1, landkreisdf_extyref=None):
    columns_to_add = [col for col in landkreisdf_exty1.columns
                       if col not in landkreisdf_ext.columns or col in ['LANDKREIS', 'year']]
    combined_landkreisdf = pd.merge(
        landkreisdf_ext, landkreisdf_exty1[columns_to_add],
        on=['LANDKREIS', 'year'], how='left'
    )
 
    if landkreisdf_extyref is not None:
        columns_to_add_ref = [col for col in landkreisdf_extyref.columns
                               if col not in combined_landkreisdf.columns or col in ['LANDKREIS', 'year']]
        combined_landkreisdf = pd.merge(
            combined_landkreisdf, landkreisdf_extyref[columns_to_add_ref],
            on=['LANDKREIS', 'year'], how='left'
        )
 
    return combined_landkreisdf

def get_landkreis_maincrop(landkreisdf, gld_no_geom):
    # Sum area_ha across years by LANDKREIS and kulturart
    area_sum_dict = {}

    for year, df in gld_no_geom.items():
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

    landkreisdf_crop = landkreisdf.merge(
        dominant_kulturart[['LANDKREIS', 'kulturart', 'Gruppe', 'total_area_ha']],
        on='LANDKREIS',
        how='left'
    )

    # column renames
    landkreisdf_crop = landkreisdf_crop.rename(columns={"kulturart": "main_crop",
                                                            "Gruppe": "main_crop_group",
                                                            "total_area_ha": "main_crop_totarea"})

    # df contains unique landkreis, their landkreis, predominant crop type and associated crop group

    # Check for missing values in the merged DataFrame
    missing_values = landkreisdf_crop.isnull().sum()
    print("Missing values in landkreisdf_crop:")
    print(missing_values[missing_values > 0])

    # check the first and last columns of landkreisdf_crop
    print("First columns of landkreisdf_crop:")
    print(landkreisdf_crop.columns[:10])
    print("Last columns of landkreisdf_crop:")
    print(landkreisdf_crop.columns[-10:])

    return landkreisdf_crop

def to_gdf(landkreisdf_ext):
    # Load Germany Landkreise boundaries (shapefile)
    landkreis_geoms = gpd.read_file(data_main_path + '/raw/verwaltungseinheiten/NDS_Landkreise.shp')
    landkreis_geoms.info()

    # Merge on LANDKREIS (shapefile is Landkreis-level, not CELLCODE-level)
    landkreisgdf = landkreisdf_ext.merge(landkreis_geoms, on='LANDKREIS')
    # Convert the DataFrame to a GeoDataFrame
    landkreisgdf = gpd.GeoDataFrame(landkreisgdf, geometry='geometry')

    return landkreis_geoms, landkreisgdf

def create_fulllandkreisdf(return_mode='without_small', gld_no_geom=None, ref_year=2016):
    output_dir = data_main_path + '/interim/landkreisdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    landkreisdf_filename = os.path.join(output_dir, f'landkreisdf_{return_mode}_withref{ref_year}.parquet')
 
    if os.path.exists(landkreisdf_filename):
        landkreisdf_full = pd.read_parquet(landkreisdf_filename)
        print(f"Loaded landkreisdf from {landkreisdf_filename}")
    else:
        if gld_no_geom is None:
            print("Loading gld_no_geom with process_nogeom_data()")
            gld_no_geom = fd.process_nogeom_data(return_mode=return_mode)
        else:
            print("Using provided gld_no_geom")
 
        landkreisdf_allyears = {}
        for key, gld in gld_no_geom.items():
            landkreisdf_yearly = create_landkreisdf_base(gld)
            check_duplicates_landkreis(landkreisdf_yearly)
            landkreisdf_allyears[key] = landkreisdf_yearly
 
        landkreisdf_base = pd.concat(landkreisdf_allyears.values(), ignore_index=True)
 
        landkreisdf_ydiff = calculate_yearlydiff_landkreis(landkreisdf_base)
        landkreisdf_exty1 = calculate_diff_fromy1_landkreis(landkreisdf_base)
        landkreisdf_extyref = calculate_diff_fromyear(landkreisdf_base, group_col='LANDKREIS', ref_year=ref_year)
        landkreisdf_ext = combine_landkreisdfs(landkreisdf_ydiff, landkreisdf_exty1, landkreisdf_extyref)
 
        for column in landkreisdf_ext.columns:
            infinite_values = landkreisdf_ext[column].isin([np.inf, -np.inf])
            print(f"Infinite values present in {column}:", infinite_values.any())
 
            if infinite_values.any():
                print(f"Rows with infinite values in {column}:")
                print(landkreisdf_ext[infinite_values])
 
            landkreisdf_ext[column].replace([np.inf, -np.inf], np.nan, inplace=True)
 
        landkreisdf = landkreisdf_ext
        landkreisdf_full = get_landkreis_maincrop(landkreisdf, gld_no_geom)
        landkreisdf_full.to_parquet(landkreisdf_filename)
        print(f"Saved landkreisdf to {landkreisdf_filename}")
 
        del gld_no_geom, landkreisdf_allyears, landkreisdf_base, landkreisdf_ydiff, landkreisdf_exty1, landkreisdf_extyref, landkreisdf_ext
        gc.collect()
 
    return landkreisdf_full


# %% drop landkreisdf outliers
def clean_landkreisdf(landkreisdf):
    # (kept analogous to the CELLCODE version's manual coastal/edge exclusions)
    landkreisdf_cl = landkreisdf[~landkreisdf['LANDKREIS'].isin(
        ["Küstenmeer Region Lüneburg", "Küstenmeer Region Weser-Ems"]
    )]

    logging.info(f"Final cleaned landkreisdf shape: {landkreisdf_cl.shape}")

    return landkreisdf_cl







# %% B.
#########################################################################
# compute mean and median for columns in landkreisdf. save the results to a csv file
def desc_landkreis(landkreisdf, ref_year=2016):
    def compute_landkreis_allyear_stats(landkreisdf):
        landkreis_allyears_stats = landkreisdf.select_dtypes(include='number').describe()
        landkreis_allyears_stats['statistic'] = landkreis_allyears_stats.index
        landkreis_allyears_stats = landkreis_allyears_stats[
            ['statistic'] + list(landkreis_allyears_stats.columns[:-1])
        ]
 
        output_dir = 'reports/statistics'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'landkreis_allyears_stats.csv')
        if not os.path.exists(filename):
            landkreis_allyears_stats.to_csv(filename, index=False)
            print(f"Saved gen_stats to {filename}")
 
        return landkreis_allyears_stats
    landkreis_allyears_stats = compute_landkreis_allyear_stats(landkreisdf)
 
    def compute_landkreis_year_average(landkreisdf, ref_year):
        landkreis_yearly_stats = landkreisdf.groupby('year').agg(
            fields_sum=('fields', 'sum'),
            fields_mean=('fields', 'mean'),
            fields_std=('fields', 'std'),
            fields_av_yearlydiff=('fields_yearly_diff', 'mean'),
            fields_adiffy1=('fields_diff_from_y1', 'mean'),
            fields_apercdiffy1=('fields_percdiff_to_y1', 'mean'),
            **{f'fields_adiffy{ref_year}': (f'fields_diff_from_y{ref_year}', 'mean')},
            **{f'fields_apercdiffy{ref_year}': (f'fields_percdiff_to_y{ref_year}', 'mean')},
 
            group_count_mean=('group_count', 'mean'),
            group_count_av_yearlydiff=('group_count_yearly_diff', 'mean'),
            group_count_adiffy1=('group_count_diff_from_y1', 'mean'),
            group_count_apercdiffy1=('group_count_percdiff_to_y1', 'mean'),
            **{f'group_count_adiffy{ref_year}': (f'group_count_diff_from_y{ref_year}', 'mean')},
            **{f'group_count_apercdiffy{ref_year}': (f'group_count_percdiff_to_y{ref_year}', 'mean')},
 
            areakm2_sum_sum=('areakm2_sum', 'sum'),
            areakm2_sum_mean=('areakm2_sum', 'mean'),
            areakm2_sum_std=('areakm2_sum', 'std'),
            areakm2_sum_av_yearlydiff=('areakm2_sum_yearly_diff', 'mean'),
            areakm2_sum_adiffy1=('areakm2_sum_diff_from_y1', 'mean'),
            areakm2_sum_apercdiffy1=('areakm2_sum_percdiff_to_y1', 'mean'),
            **{f'areakm2_sum_adiffy{ref_year}': (f'areakm2_sum_diff_from_y{ref_year}', 'mean')},
            **{f'areakm2_sum_apercdiffy{ref_year}': (f'areakm2_sum_percdiff_to_y{ref_year}', 'mean')},
 
            mareakm2_mean=('mareakm2', 'mean'),
            mareakm2_std=('mareakm2', 'std'),
            mareakm2_av_yearlydiff=('mareakm2_yearly_diff', 'mean'),
            mareakm2_adiffy1=('mareakm2_diff_from_y1', 'mean'),
            mareakm2_apercdiffy1=('mareakm2_percdiff_to_y1', 'mean'),
            **{f'mareakm2_adiffy{ref_year}': (f'mareakm2_diff_from_y{ref_year}', 'mean')},
            **{f'mareakm2_apercdiffy{ref_year}': (f'mareakm2_percdiff_to_y{ref_year}', 'mean')},
 
            medareakm2_mean=('medareakm2', 'mean'),
            medareakm2_med=('medareakm2', 'median'),
            medareakm2_std=('medareakm2', 'std'),
            medareakm2_av_yearlydiff=('medareakm2_yearly_diff', 'mean'),
            medareakm2_adiffy1=('medareakm2_diff_from_y1', 'mean'),
            medareakm2_apercdiffy1=('medareakm2_percdiff_to_y1', 'mean'),
            medareakm2_yearlydiff_med=('medareakm2_yearly_diff', 'median'),
            medareakm2_diffy1_med=('medareakm2_diff_from_y1', 'median'),
            medareakm2_percdiffy1_med=('medareakm2_percdiff_to_y1', 'median'),
            **{f'medareakm2_adiffy{ref_year}': (f'medareakm2_diff_from_y{ref_year}', 'mean')},
            **{f'medareakm2_apercdiffy{ref_year}': (f'medareakm2_percdiff_to_y{ref_year}', 'mean')},
            **{f'medareakm2_diffy{ref_year}_med': (f'medareakm2_diff_from_y{ref_year}', 'median')},
            **{f'medareakm2_percdiffy{ref_year}_med': (f'medareakm2_percdiff_to_y{ref_year}', 'median')},
 
            mperi_mean=('mperi', 'mean'),
            mperi_std=('mperi', 'std'),
            mperi_av_yearlydiff=('mperi_yearly_diff', 'mean'),
            mperi_adiffy1=('mperi_diff_from_y1', 'mean'),
            mperi_apercdiffy1=('mperi_percdiff_to_y1', 'mean'),
            **{f'mperi_adiffy{ref_year}': (f'mperi_diff_from_y{ref_year}', 'mean')},
            **{f'mperi_apercdiffy{ref_year}': (f'mperi_percdiff_to_y{ref_year}', 'mean')},
 
            medperi_med=('medperi', 'median'),
            medperi_yearlydiff_med=('medperi_yearly_diff', 'median'),
            medperi_diffy1_med=('medperi_diff_from_y1', 'median'),
            medperi_percdiffy1_med=('medperi_percdiff_to_y1', 'median'),
            **{f'medperi_diffy{ref_year}_med': (f'medperi_diff_from_y{ref_year}', 'median')},
            **{f'medperi_percdiffy{ref_year}_med': (f'medperi_percdiff_to_y{ref_year}', 'median')},
 
            mshape_mean=('mshape', 'mean'),
            mshape_std=('mshape', 'std'),
            mshape_av_yearlydiff=('mshape_yearly_diff', 'mean'),
            mshape_adiffy1=('mshape_diff_from_y1', 'mean'),
            mshape_apercdiffy1=('mshape_percdiff_to_y1', 'mean'),
            **{f'mshape_adiffy{ref_year}': (f'mshape_diff_from_y{ref_year}', 'mean')},
            **{f'mshape_apercdiffy{ref_year}': (f'mshape_percdiff_to_y{ref_year}', 'mean')},
 
            medshape_mean=('medshape', 'mean'),
            medshape_std=('medshape', 'std'),
            medshape_av_yearlydiff=('medshape_yearly_diff', 'mean'),
            medshape_adiffy1=('medshape_diff_from_y1', 'mean'),
            medshape_apercdiffy1=('medshape_percdiff_to_y1', 'mean'),
            medshape_med=('medshape', 'median'),
            medshape_yearlydiff_med=('medshape_yearly_diff', 'median'),
            medshape_diffy1_med=('medshape_diff_from_y1', 'median'),
            medshape_percdiffy1_med=('medshape_percdiff_to_y1', 'median'),
            **{f'medshape_adiffy{ref_year}': (f'medshape_diff_from_y{ref_year}', 'mean')},
            **{f'medshape_apercdiffy{ref_year}': (f'medshape_percdiff_to_y{ref_year}', 'mean')},
            **{f'medshape_diffy{ref_year}_med': (f'medshape_diff_from_y{ref_year}', 'median')},
            **{f'medshape_percdiffy{ref_year}_med': (f'medshape_percdiff_to_y{ref_year}', 'median')},
 
            fields_perarea_mean=('fields_perarea', 'mean'),
            fields_perarea_med=('fields_perarea', 'median'),
            fields_perarea_std=('fields_perarea', 'std'),
            fields_perarea_av_yearlydiff=('fields_perarea_yearly_diff', 'mean'),
            fields_perarea_adiffy1=('fields_perarea_diff_from_y1', 'mean'),
            fields_perarea_diffy1_med=('fields_perarea_diff_from_y1', 'median'),
            fields_perarea_apercdiffy1=('fields_perarea_percdiff_to_y1', 'mean'),
            fields_perarea_percdiffy1_med=('fields_perarea_percdiff_to_y1', 'median'),
            **{f'fields_perarea_adiffy{ref_year}': (f'fields_perarea_diff_from_y{ref_year}', 'mean')},
            **{f'fields_perarea_diffy{ref_year}_med': (f'fields_perarea_diff_from_y{ref_year}', 'median')},
            **{f'fields_perarea_apercdiffy{ref_year}': (f'fields_perarea_percdiff_to_y{ref_year}', 'mean')},
            **{f'fields_perarea_percdiffy{ref_year}_med': (f'fields_perarea_percdiff_to_y{ref_year}', 'median')},
 
        ).reset_index()
 
        return landkreis_yearly_stats
    landkreis_yearly_stats = compute_landkreis_year_average(landkreisdf, ref_year)
 
    return landkreis_allyears_stats, landkreis_yearly_stats
 

# Silence the print statements in a function call
def silence_prints(func, *args, **kwargs):
    # Create a string IO stream to catch any print outputs
    with io.StringIO() as f, contextlib.redirect_stdout(f):
        return func(*args, **kwargs)  # Call the function without print outputs
######################################################################################

