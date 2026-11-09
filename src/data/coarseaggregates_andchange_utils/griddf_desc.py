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

''' This script contains functions for:
    - modifying gld to contain kulturcode descriptions.
    - creating griddf and clean griddf (having removed assumed outliers).
    - computing descriptive statistics for griddf.
'''

# %% B. Creating griddf
def create_griddf_base(gld):
    """
    Create a griddf dataFrame with aggregated statistics that summarize field data.

    Parameters:
    gld (dataFrame): Input dataFrame with columns ['CELLCODE', 'year', 'LANDKREIS'] 
                        and additional numeric columns for aggregation.

    Returns:
    A  DataFrame with unique CELLCODE, year, and LANDKREIS rows,
                  enriched with aggregated fields.
    """
    
    required_columns = ['CELLCODE', 'year', 'LANDKREIS', 'row_id',\
        'Gruppe', 'area_km2', 'peri_km', 'shape']
    missing = [col for col in required_columns if col not in gld.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
    
    # Step 1: Create base griddf
    columns = ['CELLCODE', 'year', 'LANDKREIS']
    griddf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created griddf with shape {griddf.shape}")
    logging.info(f"Columns in griddf: {griddf.columns}")

    # Helper function for aggregation
    def add_aggregated_column(griddf, gld, column, aggfunc, new_col):
        logging.info(f"Adding column '{new_col}' using '{aggfunc}' on '{column}'.")
        temp = gld.groupby(['CELLCODE', 'year'])[column].agg(aggfunc).reset_index()
        temp.columns = ['CELLCODE', 'year', new_col]
        return pd.merge(griddf, temp, on=['CELLCODE', 'year'], how='left')

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
        griddf = add_aggregated_column(griddf, gld, agg['column'], agg['aggfunc'], agg['new_col'])

    # Rate of fields per hectare of land per grid
    griddf['fields_perarea'] = (griddf['fields'] / griddf['areakm2_sum'])

    return griddf


# check for duplicates in the griddf
def check_duplicates(griddf):
    duplicates = griddf[griddf.duplicated(subset=['CELLCODE', 'year'], keep=False)]
    print(f"Number of duplicates in griddf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")
            
#yearly gridcell differences and differences from first year
def calculate_yearlydiff(griddf): #yearly gridcell differences
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()
    
    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = griddf_ext.groupby('CELLCODE')[col].diff().fillna(0)
    # Calculate yearly relative difference for numeric columns and store in the dictionary
        new_columns[f'{col}_yearly_percdiff'] = (griddf_ext.groupby('CELLCODE')[col].diff() / griddf_ext.groupby('CELLCODE')[col].shift(1)).fillna(0) * 100
    
    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_ext = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_ext    


# %%
def calculate_diff_fromy1(griddf): #yearly differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()

    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Get the first occurrence of each unique CELLCODE in the griddf_ext DataFrame. 
    y1_df = griddf_ext.groupby('CELLCODE').first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[['CELLCODE'] + list(numeric_columns)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    # Merge the first year values back into the original DataFrame
    griddf_ext = pd.merge(griddf_ext, y1_df, on='CELLCODE', how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = griddf_ext[col] - griddf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((griddf_ext[col] - griddf_ext[f'{col}_y1']) / griddf_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    griddf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_exty1 = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_exty1

# %%
def calculate_diff_fromyear(df, group_col='CELLCODE', ref_year=2016):
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

# %% 
def combine_griddfs(griddf_ext, griddf_exty1, griddf_extyref=None):
    # Ensure the merge is based on 'CELLCODE' and 'year'
    columns_to_add = [col for col in griddf_exty1.columns if col not in griddf_ext.columns or col in ['CELLCODE', 'year']]
    combined_griddf = pd.merge(griddf_ext, griddf_exty1[columns_to_add], on=['CELLCODE', 'year'], how='left')
 
    if griddf_extyref is not None:
        columns_to_add_ref = [col for col in griddf_extyref.columns if col not in combined_griddf.columns or col in ['CELLCODE', 'year']]
        combined_griddf = pd.merge(combined_griddf, griddf_extyref[columns_to_add_ref], on=['CELLCODE', 'year'], how='left')
 
    return combined_griddf


# %%
def to_gdf(griddf_ext):
    # Load Germany grid_landkreise to obtain the geometry
    with open(data_main_path+'/interim/grid_landkreise.pkl', 'rb') as f:
        grid_geoms = pickle.load(f)
    grid_geoms.info()
    
    gridgdf = griddf_ext.merge(grid_geoms, on='CELLCODE')
    # Convert the DataFrame to a GeoDataFrame
    gridgdf = gpd.GeoDataFrame(gridgdf, geometry='geometry')
    # Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
    gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
    gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)

    
    return grid_geoms, gridgdf

# %%
def create_fullgriddf(return_mode='without_small', gld_no_geom=None, ref_year=2016):
    output_dir = data_main_path+'/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    # Filename depends on return_mode and ref_year, so each version is cached separately
    griddf_filename = os.path.join(output_dir, f'griddf_{return_mode}_ref{ref_year}.parquet')
 
    if os.path.exists(griddf_filename):
        griddf = pd.read_parquet(griddf_filename)
        print(f"Loaded griddf from {griddf_filename}")
    else:
        if gld_no_geom is None:
            print("Loading gld_no_geom with process_nogeom_data()")
            gld_no_geom = fd.process_nogeom_data(return_mode=return_mode)
        else:
            print("Using provided gld_no_geom")
 
        # create a griddf_base for each year
        griddf_allyears = {}
        for key, gld in gld_no_geom.items():
            griddf_yearly = create_griddf_base(gld)
            check_duplicates(griddf_yearly)
            griddf_allyears[key] = griddf_yearly
 
        # put all years' griddfs into one dataframe
        griddf_base = pd.concat(griddf_allyears.values(), ignore_index=True)
 
        # calculate differences
        griddf_ydiff = calculate_yearlydiff(griddf_base)
        griddf_exty1 = calculate_diff_fromy1(griddf_base)
        griddf_extyref = calculate_diff_fromyear(griddf_base, group_col='CELLCODE', ref_year=ref_year)
        griddf_ext = combine_griddfs(griddf_ydiff, griddf_exty1, griddf_extyref)
 
        # Check for infinite values in all columns
        for column in griddf_ext.columns:
            infinite_values = griddf_ext[column].isin([np.inf, -np.inf])
            print(f"Infinite values present in {column}:", infinite_values.any())
 
            if infinite_values.any():
                print(f"Rows with infinite values in {column}:")
                print(griddf_ext[infinite_values])
 
            griddf_ext[column].replace([np.inf, -np.inf], np.nan, inplace=True)
 
        griddf = griddf_ext
        griddf.to_parquet(griddf_filename)
        print(f"Saved griddf to {griddf_filename}")
 
        del gld_no_geom, griddf_allyears, griddf_base, griddf_ydiff, griddf_exty1, griddf_extyref, griddf_ext
        gc.collect()
 
    return griddf


# %% drop griddf outliers i.e., grids which have  fields < 300
# but only if all fields < 300 for all years in which the grid is in the dataset
def clean_griddf(griddf):
    # Remove grids with fields < 300
    griddf_clean = griddf[~(griddf['fields'] < 300)]
    outliers = griddf[griddf['fields'] < 300]
    # Log the count of unique values of 'CELLCODE' in the outliers
    logging.info(f"Unique CELLCODES with fields < 300: {outliers['CELLCODE'].nunique()}")

    # Step 2: Create a DataFrame for unique CELLCODES
    # and total count of years for unique CELLOCODES in griddf and outliers
    griddf_unique = griddf.groupby('CELLCODE').agg(total_occurrence=('year', 'count')).reset_index()
    outliers_unique = outliers.groupby('CELLCODE').agg(total_occurrence=('year', 'count')).reset_index()

    # Step 3: Add a column to check if occurrences match
    merged_outliers = outliers_unique.merge(
        griddf_unique, 
        on='CELLCODE', 
        suffixes=('_outliers', '_griddf'),
        how='left'
    )
    merged_outliers['all_years_in_data'] = merged_outliers['total_occurrence_outliers'] == merged_outliers['total_occurrence_griddf']

    # Step 4: Filter out rows where 'all_years_in_data' is no
    unmatched_outliers = merged_outliers[merged_outliers['all_years_in_data'] == False]

    # Step 5: Filter original outliers DataFrame for unmatched CELLCODES
    unmatched_outlier_codes = unmatched_outliers['CELLCODE']
    unmatched_outliers_df = outliers[outliers['CELLCODE'].isin(unmatched_outlier_codes)]

    # Step 6: Join these rows to griddf_clean
    final_cleaned_griddf = pd.concat([griddf_clean, unmatched_outliers_df], ignore_index=True)
    
    # Step 7: Create a final outliers DataFrame without the unmatched CELLCODES
    final_outliers = outliers[~(outliers['CELLCODE'].isin(unmatched_outlier_codes))]
    
    # Step 8: Drop rows with specific CELLCODEs and LANDKREIS
    # drop griddf_cl rows where ['CELLCODE'].isin(['10kmE431N333', '10kmE431N334'])
    griddf_cl = final_cleaned_griddf.copy()
    griddf_cl = griddf_cl[~griddf_cl['CELLCODE'].isin(['10kmE431N333', '10kmE431N334', '10kmE435N317', '10kmE436N317'])]
    
    # drop griddf_cl rows where ['LANDKREIS'].isin(['Küstenmeer Region Lüneburg', 'Küstenmeer Region Weser-Ems'])
    griddf_cl = griddf_cl[~griddf_cl['LANDKREIS'].isin(["Küstenmeer Region Lüneburg", "Küstenmeer Region Weser-Ems"])]
    
    logging.info(f"Final cleaned griddf shape: {griddf_cl.shape}")
   

    return griddf_cl, final_outliers

# %% B.
#########################################################################
# compute mean and median for columns in griddf. save the results to a csv file
def desc_grid(griddf, ref_year=2016):
    def compute_grid_allyear_stats(griddf):
        grid_allyears_stats = griddf.select_dtypes(include='number').describe()
        grid_allyears_stats['statistic'] = grid_allyears_stats.index
        grid_allyears_stats = grid_allyears_stats[['statistic'] + list(grid_allyears_stats.columns[:-1])]
 
        output_dir = 'reports/statistics'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'grid_allyears_stats.csv')
        if not os.path.exists(filename):
            grid_allyears_stats.to_csv(filename, index=False)
            print(f"Saved gen_stats to {filename}")
 
        return grid_allyears_stats
    grid_allyears_stats = compute_grid_allyear_stats(griddf)
 
    def compute_grid_year_average(griddf, ref_year):
        grid_yearly_stats = griddf.groupby('year').agg(
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
 
        return grid_yearly_stats
    grid_yearly_stats = compute_grid_year_average(griddf, ref_year)
 
    return grid_allyears_stats, grid_yearly_stats

# Silence the print statements in a function call
def silence_prints(func, *args, **kwargs):
    # Create a string IO stream to catch any print outputs
    with io.StringIO() as f, contextlib.redirect_stdout(f):
        return func(*args, **kwargs)  # Call the function without print outputs
######################################################################################

