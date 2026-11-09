# %%
from pathlib import Path
import os, re, gc, shutil, zipfile
import pandas as pd
import numpy as np
import chardet
import copy
import pdfplumber
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "lower_saxony_fisc": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root = os.getcwd()
data_main_path = open(project_root + "/datapath.txt").read()

from src.data.dataprocessing_utils import dataload as dl

#  Required directories:
os.makedirs("reports/Kulturcode", exist_ok=True)

# PART A:
# ================================
# Extract the unique kulturcodes from the gld data
# and load the kulturart from csv and pdf files.
# ================================
# Extract unique kulturcodes from the gld data and save them to an Excel file.
def get_kultucodes():
    output_dir = 'reports/Kulturcode/kulturcode_act.xlsx'
    output_image_path='reports/Kulturcode/unique_kulturcode_counts_by_year.png'
    gld_paths = dl.load_data(loadExistingData=True)
    
    years = range(2012, 2025)
    kulturcode_act = {}  # Will store year → DataFrame of unique kulturcodes
    
    with pd.ExcelWriter(output_dir, engine='openpyxl') as writer:
        for year in years:
            # Load the gld data for the current year
            gld_path = gld_paths[year]
            gld = gpd.read_parquet(gld_path)
            print(f"{year}: CRS of data: EPSG:{gld.crs.to_epsg()}")

            # Get unique values of 'kulturcode' for this year
            unique_codes = gld['kulturcode'].unique()
            
            # Convert to DataFrame immediately
            df = pd.DataFrame(unique_codes, columns=['kulturcode'])
            kulturcode_act[year] = df  # Store DataFrame in dictionary

            # Write to Excel sheet
            df.to_excel(writer, sheet_name=str(year), index=False)
            print(f"Processed year {year} with {len(unique_codes)} unique kulturcodes.")

    print(f"Unique kulturcodes by year have been written to '{output_dir}'")
    
    # Make a plot of the kulturcode counts by year
    # Create a DataFrame to store the year and the count of unique 'kulturcode' values
    df_year = {
        'year': list(kulturcode_act.keys()),
        'unique_kulturcode_count': [len(kulturcodes) for kulturcodes in kulturcode_act.values()]
    }
    df_unique_kulturcodes = pd.DataFrame(df_year)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df_unique_kulturcodes['year'], df_unique_kulturcodes['unique_kulturcode_count'], marker='o')
    plt.title('Unique Kulturcode Counts by Year')
    plt.xlabel('Year')
    plt.ylabel('Unique Kulturcode Count')
    plt.grid(False)
    plt.savefig(output_image_path)
    plt.show()   
    
    return kulturcode_act

# load CSV and PDF files with kulturcodes and get their descriptions.
def get_kulturcode_desc(data_main_path, save_csv=True, load_existing=True):
    """
    Extracts, processes, and returns a dictionary of DataFrames keyed by year.
    Optionally saves or loads from already processed CSV outputs.

    Parameters
    ----------
    data_main_path : str
        Path to the root folder containing the /raw directory
    save_csv : bool, default True
        If True, saves each year's processed DataFrame to CSV
    load_existing : bool, default True
        If True, loads already processed CSV instead of re-processing

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary where keys are years and values are processed DataFrames
    """

    # --- Folder setup ---
    base_dir = os.path.join(data_main_path, "raw")
    schlaege_dir = os.path.join(base_dir, "nieder_origin/schlaege")
    kultur_dir = os.path.join(base_dir, "nieder_origin/kulturcodes")
    kultur_ori_dir = os.path.join(kultur_dir, "kultur_orifiles")
    kultur_csv_dir = os.path.join(kultur_dir, "kultur_csvfiles")

    # Ensure kultur directories exist. Schlaege directory must already exist with dataload
    # If schlaege_dir does not exist, you need to run the dataload script first.
    # or the get_kultucodes function which relies on dataload module.
    os.makedirs(kultur_ori_dir, exist_ok=True)
    os.makedirs(kultur_csv_dir, exist_ok=True)

    # --- Year records ---
    kc_info = [
        ['year', 'zipped_folder', 'kulturart'],
        [2012, 'schlaege_skizzen_2012.zip', 'Kulturart.csv'],
        [2013, 'schlaege_skizzen_2013.zip', 'Kulturart.csv'],
        [2014, 'schlaege_skizzen_2014.zip', 'Kulturcodes_BNK_2014.pdf'],
        [2015, 'schlaege_skizzen_2015.zip', 'Kulturcodes_BNK_2015.pdf'],
        [2016, 'schlaege_hauptzahlung_2016.zip', 'Kulturcodes_BNK_2016.pdf'],
        [2017, 'schlaege_hauptzahlung_2017.zip', 'Kulturcodes_BNK_2017.pdf'],
        [2018, 'schlaege_hauptzahlung_2018.zip', 'Kulturcodes_BNK_2018_02.pdf'],
        [2019, 'schlaege_hauptzahlung_2019.zip', 'Kulturcodes_BNK_2019_04.pdf'],
        [2020, 'schlaege_hauptzahlung_2020.zip', 'Kulturcodes_BNK_2020_03.pdf'],
        [2021, 'schlaege_hauptzahlung_2021.zip', 'Kulturcodes_BNK_2021_03.pdf'],
        [2022, 'schlaege_hauptzahlung_2022.zip', 'Kulturcodes_BNK_2022_04.pdf'],
        [2023, 'schlaege_hauptzahlung_2023.zip', 'Kulturcodes_BNK_2023_08.pdf'],
        [2024, 'schlaege_hauptzahlung_2024.zip', 'Kulturcodes_BNK_2024_10.pdf']
    ]
    records = [{"year": row[0], "zip": row[1], "kultur": row[2]} for row in kc_info[1:]]

    kulturcode_dict = {}

    # --- Main loop ---
    for record in records:
        year = record["year"]
        kultur_file = record["kultur"]
        final_csv_path = os.path.join(kultur_csv_dir, f"kulturcodes_{year}.csv")

        # --- NEW: load from existing CSV if allowed ---
        if load_existing and os.path.exists(final_csv_path):
            try:
                df = pd.read_csv(final_csv_path, encoding='utf-8-sig')
                kulturcode_dict[year] = df
                print(f"[{year}] Loaded from existing CSV: {final_csv_path}")
                continue
            except Exception as e:
                print(f"[{year}] Failed to load existing CSV, will re-process: {e}")

        # === Extraction & Processing ===
        zip_path = os.path.join(schlaege_dir, record["zip"])
        file_path = os.path.join(kultur_ori_dir, f"{year}_{os.path.basename(kultur_file)}")

        if not os.path.exists(file_path):
            if not os.path.exists(zip_path):
                print(f"⚠️ Zip file for {year} not found: {zip_path}")
                continue
            with zipfile.ZipFile(zip_path, 'r') as z:
                if kultur_file in z.namelist():
                    temp_dir = os.path.join(kultur_ori_dir, f"temp_{year}")
                    os.makedirs(temp_dir, exist_ok=True)
                    z.extract(kultur_file, temp_dir)
                    shutil.move(os.path.join(temp_dir, kultur_file), file_path)
                    shutil.rmtree(temp_dir)
                    print(f"  Extracted {year}_{os.path.basename(kultur_file)}")
                else:
                    print(f"⚠️ {kultur_file} not found in {zip_path}")
                    continue

        try:
            # --- Load raw file ---
            if file_path.endswith(".csv"):
                with open(file_path, 'rb') as f:
                    enc_guess = chardet.detect(f.read())['encoding']
                df = pd.read_csv(file_path, encoding=enc_guess, sep=None, engine='python')
                df = df.iloc[:, :2]
            elif file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    all_rows = []
                    for page in pdf.pages:
                        table = page.extract_table()
                        if table:
                            all_rows.extend(table)
                if len(all_rows) > 1:
                    max_cols = max(len(row) for row in all_rows)
                    cleaned_rows = [row + [''] * (max_cols - len(row)) for row in all_rows[1:]]
                    df = pd.DataFrame(cleaned_rows)
                else:
                    df = pd.DataFrame()
            else:
                continue

            # === CLEANING and YEAR-SPECIFIC RULES ===
            df.dropna(axis=1, how='all', inplace=True)
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.map(lambda x: np.nan if str(x).strip() in ['0','1','2','3'] else x)
            df.dropna(axis=1, how='all', inplace=True)

            if year in range(2014, 2025):
                mask = df.iloc[:, 0].isna() & df.iloc[:, 1].notna()
                df.loc[mask, df.columns[0]] = df.loc[mask, df.columns[1]]
                df.loc[mask, df.columns[1]] = np.nan
                df.columns = range(df.shape[1])
                moved = True
                while moved:
                    moved = False
                    for i in range(1, len(df.columns) - 1):
                        left_col, right_col = df.columns[i], df.columns[i + 1]
                        mask = df[left_col].isna() & df[right_col].notna()
                        if mask.any():
                            moved = True
                            df.loc[mask, left_col] = df.loc[mask, right_col]
                            df.loc[mask, right_col] = np.nan

            # Normalise final output
            df = df.iloc[:, :2]
            df.columns = ['Code', 'Kulturart']
            df = df[df['Code'].notna() & (df['Code'] != '')].reset_index(drop=True)

            # Add Gruppe column
            gruppe_mask = df['Code'].astype(str).str.contains('Gruppe', na=False)
            df['Gruppe'] = np.nan
            df.loc[gruppe_mask, 'Gruppe'] = df.loc[gruppe_mask, 'Kulturart']
            df['Gruppe'] = df['Gruppe'].ffill().apply(
                lambda x: re.sub(r'[:\*]', '', str(x)).strip() if pd.notna(x) else x
            )

            # Code = first 3 digits only
            df['Code'] = df['Code'].apply(lambda v: ''.join(re.findall(r'\d', str(v))[:3]) if re.findall(r'\d', str(v)) else np.nan)
            df = df[df['Code'].notna() & (df['Code'] != '')].reset_index(drop=True)

            kulturcode_dict[year] = df

            if save_csv:
                df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
                print(f"[{year}] Saved to {final_csv_path}")

        except Exception as e:
            print(f"[{year}] Error processing: {e}")
            
    # print head for a sample year e.g. 2020
    if 2020 in kulturcode_dict:
        print(f"Sample data for 2020:\n{kulturcode_dict[2020].head()}")
    else:
        print("No data for 2020 found in kulturcode_dict.")       
    # Ensure keys are integers and rename columns
    for key in kulturcode_dict:
    # rename columns Code and Kulturart to 'kulturcode' and 'kulturart'
        kulturcode_dict[key].rename(columns={'Code': 'kulturcode', 'Kulturart': 'kulturart'}, inplace=True)
    # Check for non-numeric kulturcode values
        non_numeric_kulturcodes = [code for code in kulturcode_dict[key]['kulturcode'] if not str(code).replace('.', '', 1).isdigit()]
        if non_numeric_kulturcodes:
            print(f"{key}: Non-numeric kulturcode values found: {non_numeric_kulturcodes}")
        else:
            print(f"{key}: All kulturcode values are numeric.")
        
    kulturcode_dict = {int(key): value for key, value in kulturcode_dict.items()}
    print(kulturcode_dict.keys())
    
    return kulturcode_dict


# PART B:
# ================================
# Step 1. Harmonization Functions
# ================================

def copy_kulturart(kulturart: dict) -> dict:
    """Deep copy kulturart dictionary."""
    return {year: df.copy() for year, df in kulturart.items()}

def fill_gruppe_backwards(data_dict, target_year, manual_fixes=None):
    """
    Fill missing 'Gruppe' values in target_year DataFrame by looking back through
    previous years (target_year-1, target_year-2, ...) until earliest year available.
    Optionally, apply manual fixes for specific kulturcodes.

    Parameters
    ----------
    data_dict : dict
        Dictionary with years as keys and DataFrames as values.
    target_year : int
        The year to update with missing Gruppe values.
    manual_fixes : dict, optional
        Dictionary of {kulturcode: Gruppe} to apply after backward fill.

    Returns
    -------
    data_dict : dict
        Updated dictionary with filled Gruppe values in target_year.
    missing_codes : set
        Set of kulturcodes that remain without Gruppe after searching all years and applying fixes.
    """
    df_target = data_dict[target_year].copy()
    df_target["Gruppe"] = df_target["Gruppe"].astype("string")

    # get all years less than target, sorted descending (most recent first)
    previous_years = sorted([y for y in data_dict.keys() if y < target_year], reverse=True)

    # backward fill from previous years
    for year in previous_years:
        missing_mask = df_target["Gruppe"].isna()
        if not missing_mask.any():
            break
        
        df_source = data_dict[year]
        gruppe_map = (
            df_source.dropna(subset=["Gruppe"])
                     .drop_duplicates(subset=["kulturcode"], keep="first")
                     .set_index("kulturcode")["Gruppe"]
        )
        df_target.loc[missing_mask, "Gruppe"] = (
            df_target.loc[missing_mask, "kulturcode"].map(gruppe_map).astype("string")
        )

    # apply manual fixes if provided
    if manual_fixes:
        for kode, gruppe in manual_fixes.items():
            df_target.loc[df_target["kulturcode"] == kode, "Gruppe"] = gruppe

    # Update dict
    data_dict[target_year] = df_target

    # report kulturcodes still missing
    missing_codes = set(df_target.loc[df_target["Gruppe"].isna(), "kulturcode"])
    if missing_codes:
        print(f"⚠️ Still missing Gruppe for kulturcodes in {target_year}: {missing_codes}")

    return data_dict, missing_codes


def map_gruppe(val: str) -> str:
    mapping = {
        'Greening / Landschaftselemente': 'GLK',
        'Greening': 'GLK',
        'Konditionalität (als Bindung)': 'GLK',
        'Aus der Produktion genommen': 'Aus der Produktion/Erzeugung genommen',
        'Aus der Erzeugung genommen': 'Aus der Produktion/Erzeugung genommen',
        'Aus der Produktion/Erzeugung genommen': 'Aus der Produktion/Erzeugung genommen',
        'Sonstige LF auf AL': 'Sonstige Flächen'
    }
    return mapping.get(val, val)


def apply_gruppe_mapping(kulturart_ha: dict) -> None:
    for year in kulturart_ha:
        kulturart_ha[year]['Gruppe'] = kulturart_ha[year]['Gruppe'].apply(map_gruppe)
    gc.collect()


def apply_special_mappings(kulturart_ha: dict, special_map: dict, years=(2012, 2013, 2014)) -> None:
    for year in years:
        df = kulturart_ha[year]
        for new_code, old_codes in special_map.items():
            mask = df['kulturcode'].isin(old_codes)
            df.loc[mask, 'new_kulturcode'] = new_code

            if new_code == 715:
                df.loc[mask, ['new_kulturart', 'new_Gruppe']] = ['Gemüse', 'Gemüse']
            elif new_code == 815:
                df.loc[mask, ['new_kulturart', 'new_Gruppe']] = ['Dauerkulturen', 'Dauerkulturen']

        # fill defaults
        df['new_kulturcode'] = df['new_kulturcode'].fillna(df['kulturcode']).astype('Int64')
        df['new_kulturart'] = df['new_kulturart'].fillna(df['kulturart'])
        df['new_Gruppe'] = df['new_Gruppe'].fillna(df['Gruppe'])
    gc.collect()


def range_new_kulturcode(row):
    code = int(row.get('new_kulturcode', row['kulturcode']))
    gruppe = row.get('new_Gruppe', row.get('Gruppe', ''))
    if (52 <= code <= 66) or (gruppe == 'GLK'): return 55
    if 81 <= code <= 97: return 85
    if 545 <= code <= 587: return 555
    if 590 <= code <= 595: return 595
    if 650 <= code <= 687: return 655
    if 701 <= code <= 710: return 705
    if 720 <= code <= 799: return 725
    if 801 <= code <= 806: return 805
    if 897 <= code <= 999: return 900
    return code


def apply_range_mapping(kulturart_ha: dict) -> dict:
    new_kulturart_ha = {}
    for year, df in kulturart_ha.items():
        df = df.copy()
        df['old_kulturcode'] = df['kulturcode']
        df['new_kulturcode'] = df.get('new_kulturcode', df['kulturcode'])
        df['new_kulturcode'] = df.apply(range_new_kulturcode, axis=1)
        df['new_Gruppe'] = df.get('new_Gruppe', df['Gruppe'])
        df['new_kulturart'] = df.get('new_kulturart', df['kulturart'])

        # 🔑 Ensure range-mapped codes get kulturart = Gruppe
        range_codes = {55, 85, 555, 595, 655, 705, 725, 805, 900}
        mask = df['new_kulturcode'].isin(range_codes)
        df.loc[mask, 'new_kulturart'] = df.loc[mask, 'new_Gruppe']

        new_kulturart_ha[year] = df[['old_kulturcode', 'new_kulturcode', 'new_kulturart', 'new_Gruppe']].rename(
            columns={
                'new_kulturcode': 'kulturcode',
                'new_kulturart': 'kulturart',
                'new_Gruppe': 'Gruppe'
            }
        )
        del df
        gc.collect()
    return new_kulturart_ha


# ================================
# Step 2. Multi-Gruppe Handling
# ================================

def build_inconsistent_report(new_kulturart_ha: dict, out_path: Path) -> pd.DataFrame:
    """
    Build a report of inconsistent Gruppe assignments and return
    the most recent Gruppe for all kulturcodes.

    Parameters
    ----------
    new_kulturart_ha : dict
        Dictionary of DataFrames with year as key.
    out_path : Path
        Path to save the CSV report.

    Returns
    -------
    most_recent : pd.DataFrame
        DataFrame mapping every old_kulturcode to its most recent Gruppe.
    """
    # Step 1: Concatenate all years
    all_years_list = []
    for year, df in new_kulturart_ha.items():
        temp_df = df[['kulturcode', 'Gruppe']].copy()
        temp_df['year'] = year
        all_years_list.append(temp_df)
    all_years_df = pd.concat(all_years_list, ignore_index=True)

    # Step 2: Ensure Gruppe is string and fill missing temporarily
    all_years_df['Gruppe'] = all_years_df['Gruppe'].fillna('').astype(str)

    # Step 3: Identify inconsistent codes
    group_counts = all_years_df.groupby('kulturcode')['Gruppe'].nunique()
    multi_codes = group_counts[group_counts > 1].index

    # Step 4: Pivot for inspection
    multi_df = all_years_df[all_years_df['kulturcode'].isin(multi_codes)]
    pivot_df = multi_df.pivot_table(
        index='kulturcode', columns='year', values='Gruppe', aggfunc='first'
    )
    pivot_df.columns = [f"Gruppe_{c}" for c in pivot_df.columns]

    # Step 5: Concatenate Gruppe values for inconsistent codes
    gruppe_concat = multi_df.groupby('kulturcode')['Gruppe'].apply(
        lambda v: " | ".join(sorted(set(filter(lambda x: x != '', v))))
    )

    # Step 6: Pick the most recent non-empty Gruppe for all codes
    non_empty_df = all_years_df[all_years_df['Gruppe'] != '']
    most_recent = (
        non_empty_df
        .sort_values(['kulturcode', 'year'], ascending=[True, False])
        .drop_duplicates('kulturcode')
        .set_index('kulturcode')[['Gruppe', 'year']]
        .rename(columns={'Gruppe':'Gruppe_mostrecent', 'year':'year_mostrecentGruppe'})
    )

    # Step 7: Build final report for inspection (only inconsistent codes)
    final_report = pivot_df.join(gruppe_concat).join(
        most_recent[['Gruppe_mostrecent', 'year_mostrecentGruppe']]
    )
    final_report.to_csv(out_path, encoding='utf-8-sig')

    # Cleanup
    del all_years_df, multi_df, pivot_df
    gc.collect()

    return most_recent


def enforce_most_recent_gruppe(new_kulturart_ha: dict, most_recent: pd.DataFrame) -> None:
    """
    Overwrite all Gruppe values in each year with the most recent
    non-empty assignment from most_recent mapping.
    
    We do this so that years with multiple Gruppe assignments
    will have the most recent one applied consistently.

    Parameters
    ----------
    new_kulturart_ha : dict
        Dictionary of DataFrames with year as key.
    most_recent : pd.DataFrame
        Mapping of kulturcode to its most recent Gruppe.
    """
    for year, df in new_kulturart_ha.items():
        df['Gruppe'] = df['kulturcode'].map(most_recent['Gruppe_mostrecent']).fillna('')
        # Special case correction
        df.loc[df['kulturcode'] == 860, 'Gruppe'] = 'Gemüse'
    gc.collect()



# ================================
# Step 3. Master Table
# ================================

def build_master(new_kulturart_ha: dict, years=range(2012, 2025), wide: bool = False):
    long = pd.concat(
        [df.assign(year=year)[['old_kulturcode', 'kulturcode', 'kulturart', 'Gruppe', 'year']] 
         for year, df in new_kulturart_ha.items()],
        ignore_index=True
    )

    long['old_kulturcode'] = pd.to_numeric(long['old_kulturcode'], errors='coerce').astype('Int64')
    long['kulturcode'] = pd.to_numeric(long['kulturcode'], errors='coerce').astype('Int64')
    long['kulturart'] = long['kulturart'].fillna('').astype('category')
    long['Gruppe'] = long['Gruppe'].fillna('').astype('category')

    all_kulturcode = long[['old_kulturcode','kulturcode']].drop_duplicates().reset_index(drop=True)

    latest = (long.sort_values(['kulturcode','year'])
                   .drop_duplicates('kulturcode', keep='last')
                   .loc[:, ['kulturcode','kulturart','Gruppe']]
                   .rename(columns={'kulturart':'latest_kulturart','Gruppe':'latest_Gruppe'}))

    if not wide:
        gc.collect()
        return latest.reset_index(drop=True), all_kulturcode

    # Wide mode
    years_sorted = sorted(set(long['year'].dropna().astype(int).tolist()), reverse=True)

    pivot_kulturart = (long.pivot(index='kulturcode', columns='year', values='kulturart').astype('category'))
    pivot_gruppe    = (long.pivot(index='kulturcode', columns='year', values='Gruppe').astype('category'))

    pivot_kulturart = pivot_kulturart.reindex(columns=years_sorted)
    pivot_gruppe    = pivot_gruppe.reindex(columns=years_sorted)

    pivot_kulturart.columns = [f'kulturart {y}' for y in pivot_kulturart.columns]
    pivot_gruppe.columns    = [f'Gruppe {y}'    for y in pivot_gruppe.columns]

    master = latest.set_index('kulturcode').join(pivot_kulturart, how='left').join(pivot_gruppe, how='left')

    k_cols = [f'kulturart {y}' for y in years_sorted]
    g_cols = [f'Gruppe {y}' for y in years_sorted]
    master['latest_kulturart'] = master[k_cols].apply(lambda row: next((x for x in row if pd.notna(x) and x != ''), ''), axis=1)
    master['latest_Gruppe']    = master[g_cols].apply(lambda row: next((x for x in row if pd.notna(x) and x != ''), ''), axis=1)

    del long, pivot_kulturart, pivot_gruppe
    gc.collect()

    return master.reset_index(), all_kulturcode


# ================================
# Step 4. Apply Fix File
# ================================

def apply_fix(master: pd.DataFrame, all_kulturcode: pd.DataFrame, fix_path: Path):
    kfix = pd.read_csv(fix_path, encoding='utf-8-sig').drop_duplicates('kulturcode').set_index('kulturcode')
    kfix['take_latest'] = kfix['take_latest'].fillna(0).astype(int)
    kfix['take_Gruppe'] = kfix['take_Gruppe'].fillna(0).astype(int)
    kfix['new_kulturcode'] = pd.to_numeric(kfix['new_kulturcode'], errors='coerce').astype('Int64')

    def get_actual(row):
        code = row['kulturcode']
        actual = {
            'new_kulturart_actual': row['latest_kulturart'],
            'Gruppe_actual': row['latest_Gruppe'],
            'new_kulturcode_actual': code
        }
        if code in kfix.index:
            fix = kfix.loc[code]
            if fix['take_latest']:
                actual['new_kulturart_actual'] = row['latest_kulturart']
            elif fix['take_Gruppe']:
                actual['new_kulturart_actual'] = row['latest_Gruppe']
            elif pd.notna(fix['new_kulturart']):
                actual['new_kulturart_actual'] = fix['new_kulturart']

            if pd.notna(fix['new_kulturcode']):
                actual['new_kulturcode_actual'] = int(fix['new_kulturcode'])
        return pd.Series(actual)

    master[['new_kulturart_actual','Gruppe_actual','new_kulturcode_actual']] = master.apply(get_actual, axis=1)
    

    all_kulturcode_master = all_kulturcode.merge(
        master[['kulturcode','new_kulturcode_actual', 'new_kulturart_actual','Gruppe_actual']],
        on='kulturcode',
        how='left'
    ).drop_duplicates()
    
    # rename columns
    all_kulturcode_master.rename(columns={
        'kulturcode': 'int_kulturcode',
        'new_kulturcode_actual': 'new_kulturcode',
        'new_kulturart_actual': 'kulturart',
        'Gruppe_actual': 'Gruppe'
    }, inplace=True)
    

    del kfix
    gc.collect()
    return all_kulturcode_master


# ================================
# Step 5. Validation and final fix
# ================================

def validate_master(master: pd.DataFrame) -> None:
    for col in ['new_kulturcode', 'kulturart', 'Gruppe']:
        # Check for missing (NaN) or empty string values
        missing_mask = master[col].isnull() | (master[col].astype(str).str.strip() == '')
        if missing_mask.any():
            print(f"⚠️ Missing or empty values detected in {col}: {missing_mask.sum()} rows")
        else:
            print(f"  No missing or empty values in {col}")


def report_and_fix_missing_gruppe(all_kulturcode_master: pd.DataFrame) -> pd.DataFrame:
    """
    Identify all unique kulturcodes that are missing a 'Gruppe' value,
    print their 'kulturart', and assign Gruppe if known from a predefined mapping.

    Parameters
    ----------
    all_kulturcode_master : pd.DataFrame
        DataFrame containing at least 'new_kulturcode', 'Gruppe', and 'kulturart' columns.

    Returns
    -------
    updated_df : pd.DataFrame
        DataFrame with missing Gruppe values filled where possible.
    """
    # Predefined fixes for missing Gruppe
    missing_gruppe_fix = {
        156: "Getreide",
        174: "Getreide",
        176: "Getreide",
        342: "Ölsaaten",
        412: "Ackerfutter",
        690: "Hackfrüchte",
        890: "Dauerkulturen",
        892: "Gemüse",
        896: "Zierpflanzen"
    }

    df = all_kulturcode_master.copy()
    df['Gruppe'] = df['Gruppe'].astype(str)

    # Mask for missing Gruppe (empty or NaN)
    missing_mask = df['Gruppe'].isna() | (df['Gruppe'].str.strip() == '')

    # Select unique rows with missing Gruppe
    missing_df = df.loc[missing_mask, ['new_kulturcode', 'kulturart']].drop_duplicates()

    if not missing_df.empty:
        print("Kulturcodes missing 'Gruppe' and their 'kulturart':")
        print(missing_df)
    else:
        print("No missing Gruppe values found.")

    # Apply manual fixes if available
    df['Gruppe'] = df.apply(
        lambda row: missing_gruppe_fix[row['new_kulturcode']]
        if row['new_kulturcode'] in missing_gruppe_fix else row['Gruppe'],
        axis=1
    )

    return df

# ================================
# === MAIN EXECUTION PIPELINE ===
# ================================
def process_kulturcode(data_main_path, load_existing=True):
    
    # Paths for reports and outputs
    fix_path = os.path.join(data_main_path, "raw/kulturcode_fix.csv")
    out_master = data_main_path+"/interim/kulturcode_mastermap.csv"
    group_consistency_report_path = Path("reports/Kulturcode/inconsistent_gruppe.csv")
    missing_codes_report_path = "reports/Kulturcode/missing_kulturcodes_report.csv"
    
    # Check if the CSV file already exists
    if os.path.exists(out_master):
        print("Loading kulturcode_mastermap from CSV.")
        all_kulturcode_master = pd.read_csv(out_master, encoding='utf-8-sig')
    else:
        print("Processing all_kulturcode_master.")
            
        # Step 0: Load kulturcodes and descriptions
        kulturcode_act = get_kultucodes()    
        kulturart = get_kulturcode_desc(data_main_path, load_existing)    

        # Step 1: Harmonization
        d_kulturart = copy_kulturart(kulturart)    #Copy original kulturart
        
        ## fix for 2024
        manual_fixes_2024 = {888: "AUKM", 885: "AUKM"}
        d_kulturart, missing = fill_gruppe_backwards(d_kulturart, 2024, manual_fixes=manual_fixes_2024)
        
        apply_gruppe_mapping(d_kulturart)
        special_map = {
            602: [611,612,613,614,615,619],
            603: [620],
            701: [793],
            715: list(range(710,716)),
            815: list(range(812,820)),
            833: [824],
            834: [825],
            838: [830],
            839: [831],
            860: [715],
            983: [846]
        }
        apply_special_mappings(d_kulturart, special_map)
        new_kulturart_ha = apply_range_mapping(d_kulturart)

        # Step 2: Multi-Gruppe Handling
        most_recent_mapping = build_inconsistent_report(new_kulturart_ha, group_consistency_report_path)
        enforce_most_recent_gruppe(new_kulturart_ha, most_recent_mapping)
        
        # Step 3: Build master
        kulturcode_master, all_kulturcode = build_master(new_kulturart_ha, wide=False)

        # Step 4: Apply fix
        all_kulturcode_master = apply_fix(kulturcode_master, all_kulturcode, fix_path)

        # Step 5: Validate and fix
        validate_master(all_kulturcode_master)
        all_kulturcode_master = report_and_fix_missing_gruppe(all_kulturcode_master)


        # Save final outputs   
        all_kulturcode_master.to_csv(out_master, index=False, encoding="utf-8-sig")
        print(f" Final master DataFrame saved to {out_master}")

        # Missing codes check
        missing_codes_report = {}
        master_codes = set(all_kulturcode_master['old_kulturcode'])
        for year, df in kulturcode_act.items():
            if isinstance(df, (list, tuple, np.ndarray)):
                df = pd.DataFrame(df)
            if 'kulturcode' not in df.columns:
                print(f"⚠️ Year {year} missing 'kulturcode' column.")
                continue
            df_codes = set(df['kulturcode'].dropna())
            missing = df_codes - master_codes
            if missing:
                missing_codes_report[year] = missing
                print(f"Year {year}: {len(missing)} missing codes")
            else:
                print(f"Year {year}: no missing codes")

        with open(missing_codes_report_path, "w", encoding="utf-8-sig") as f:
            f.write("year,missing_kulturcode\n")
            for year, missing_codes in missing_codes_report.items():
                for code in missing_codes:
                    f.write(f"{year},{code}\n")
        print(f"\n Missing codes report saved to {missing_codes_report_path}")
        

    return all_kulturcode_master

# %%
if __name__ == '__main__':
   kulturcode_master = process_kulturcode(data_main_path, load_existing=True)


# %%
