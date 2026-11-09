#process_farmdata.py

#############################################################
# import farm data from xml and convert to pandas dataframe
#############################################################
# %% Import necessary libraries
import zipfile
import pandas as pd
import os, re
import xml.etree.ElementTree as ET
import numpy as np
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

# %%
# Unzip the folder containing animal data
farmzip_path = data_main_path+'/raw/farmdata.zip' 
farmextract_path = data_main_path+'/raw/farmdata'

# Create the directory if it doesn't exist
if not os.path.exists(farmextract_path):
    os.makedirs(farmextract_path)   


with zipfile.ZipFile(farmzip_path, 'r') as zip_ref:
    zip_ref.extractall(farmextract_path)

# Find the XML file (assuming it's .xml)
for filename in os.listdir(farmextract_path):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(farmextract_path, filename)
        break

# Load the XML
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Register the namespace used in the file (required for correct tag searches)
namespaces = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

# Find the table rows
rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', namespaces)

data = []

# Iterate through each row and extract cell data
for row in rows:
    row_data = []
    for cell in row.findall('ss:Cell', namespaces):
        data_element = cell.find('ss:Data', namespaces)
        if data_element is not None:
            row_data.append(data_element.text)
        else:
            row_data.append('')
    data.append(row_data)

# Remove any irrelevant header/footer rows based on content
# This will depend on your data; below filters out top metadata and keeps actual data
# Adjust the slicing index as needed
data = [r for r in data if any(r)]  # Remove completely empty rows
data = data[8:]  # Skip metadata/header rows if needed

# Assume first row after slicing is header
header = data[18]
rows = data[0:]

# Create DataFrame
df = pd.DataFrame(rows, columns=header)

print(df.head())

# %% last column is missing column name because of shifted rows
# let's fix that by moving current idx 1 and 2 one column to the right
for idx in [1, 2]:
    df.iloc[idx, 1:] = df.iloc[idx, :-1].values

# %%
# Set values in row index 0
df.iloc[0, [1, 2]] = 2010
df.iloc[0, [3, 4]] = 2016
df.iloc[0, [5, 6]] = 2020

# %%
df = df.drop(index=2)

# %%
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(how='all')
df = df.reset_index(drop=True)

# %%
#copy df to a new DataFrame and drop last row containinng footer information
df_copy = df.copy()
df_copy = df_copy.iloc[:-1, :] 

# project LANDKREIS value to new column for each row
# Starting from row index 2
start_idx = 2
block_size = 20  # 1 header + 19 classes

LANDKREIS_list = []

# For rows before start_idx (0 and 1), fill None:
LANDKREIS_list.extend([None] * start_idx)

i = start_idx
while i < len(df_copy):
    LANDKREIS = df_copy.iloc[i, 0]

    rows_left = len(df_copy) - i
    num_class_rows = min(rows_left - 1, 19)  # always max 19 class rows after header

    # For the header row, add None:
    LANDKREIS_list.append(None)

    # For the 19 class rows, add LANDKREIS name:
    LANDKREIS_list.extend([LANDKREIS] * num_class_rows)

    i += 1 + num_class_rows

# Safety pad if needed:
while len(LANDKREIS_list) < len(df_copy):
    LANDKREIS_list.append(None)

df_copy['LANDKREIS'] = LANDKREIS_list

# Keep the first two rows and only rows where LANDKREIS is not None
first_two = df_copy.iloc[:2]

rest = df_copy.iloc[2:]
rest_filtered = rest[rest['LANDKREIS'].notna()]

# Concatenate back
df_copy_filtered = pd.concat([first_two, rest_filtered]).reset_index(drop=True)

df_copy_filtered.rename(columns={df_copy_filtered.columns[0]: 'Klasse'}, inplace=True)

# %% split the DataFrame into three separate DataFrames for each year
df_2010 = df_copy_filtered.iloc[:, [0, 1, 2, 7]].copy()
df_2016 = df_copy_filtered.iloc[:, [0, 3, 4, 7]].copy()
df_2020 = df_copy_filtered.iloc[:, [0, 5, 6, 7]].copy()

# rename Anzahl and LF (Landwirtschaftliche Fläche) columns
df_2010.rename(columns={df_2010.columns[1]: 'Anzahl', df_2010.columns[2]: 'LF_ha'}, inplace=True)
df_2016.rename(columns={df_2016.columns[1]: 'Anzahl', df_2016.columns[2]: 'LF_ha'}, inplace=True)
df_2020.rename(columns={df_2020.columns[1]: 'Anzahl', df_2020.columns[2]: 'LF_ha'}, inplace=True)

# create year column and insert it at index 1
df_2010.insert(1, 'year', 2010)
df_2016.insert(1, 'year', 2016)
df_2020.insert(1, 'year', 2020)

# Reorder columns: Klasse, LANDKREIS, rest
cols = ['LANDKREIS', 'Klasse'] + [col for col in df_2010.columns if col not in ['LANDKREIS', 'Klasse']]
df_2010 = df_2010[cols]
df_2016 = df_2016[cols]
df_2020 = df_2020[cols]

# keeps all rows from index 2 onward (i.e., drops rows 0 and 1)
df_2010 = df_2010.iloc[2:].reset_index(drop=True)
df_2016 = df_2016.iloc[2:].reset_index(drop=True)
df_2020 = df_2020.iloc[2:].reset_index(drop=True)

# %% Combine the three DataFrames into one
df_farm = pd.concat([df_2010, df_2016, df_2020], ignore_index=True)

# delete helper dataframes
del df_2010, df_2016, df_2020, df_copy_filtered, df_copy

# %% keep only rows where Klasse is Ingesamt
df_farmI = df_farm[df_farm['Klasse'] == 'Insgesamt'].reset_index(drop=True)

# %% check for Anzahl and LF_ha column cells that are not numeric
# if all rows in a column are numeric, we can convert them to float
# if not, we will create a new column with boolean values indicating numeric status
def is_numeric_column(column):
    """Check if all values in the column are numeric."""
    return pd.to_numeric(column, errors='coerce').notna().all()
def convert_to_numeric(column):
    """Convert column to numeric, coercing errors to NaN."""
    return pd.to_numeric(column, errors='coerce')
def check_and_convert_numeric(df, column_name):
    """Check if a column is numeric and convert it if so."""
    if is_numeric_column(df[column_name]):
        df[column_name] = convert_to_numeric(df[column_name])
    else:
        df[f"{column_name}_is_numeric"] = df[column_name].apply(lambda x: isinstance(x, (int, float)))
    return df
df_farmI = check_and_convert_numeric(df_farmI, 'Anzahl')
df_farmI = check_and_convert_numeric(df_farmI, 'LF_ha')

df_farmI.info()

# convert ha -> km² (1 km² = 100 ha) to follow unit in field analysis, then compute the mean per farm
df_farmI['LF_km2'] = df_farmI['LF_ha'] / 100
df_farmI['LF_mean'] = df_farmI['LF_km2'] / df_farmI['Anzahl']  # km² per farm

# %% sort by LANDKREIS and year
df_farmI.sort_values(by=['LANDKREIS', 'year'], inplace=True)

# %% drop row where 'LANDKREIS' contains 241001
df_farmI = df_farmI[~df_farmI['LANDKREIS'].str.contains('241001', na=False)]

# %%
# standardize LANDKREIS names
def standardize_landkreis(value):
    try:
        if pd.isna(value):
            return None
        parts = value.strip().split()
        if len(parts) < 2:
            return None
        name = ' '.join(parts[1:])
        # normalize spacing before parentheses
        name = re.sub(r'(?<!\s)\(', r' (', name)
        if ',Stadt' in name:
            name = name.replace(',Stadt', '')
            return f'Stadt {name.strip()} (kreisfrei)'
        else:
            return name.strip()
    except Exception:
        return None
df_farmI['LANDKREIS'] = df_farmI['LANDKREIS'].apply(standardize_landkreis)

# %% correct region Hannover 
df_farmI['LANDKREIS'] = df_farmI['LANDKREIS'].replace("Hannover,Region", "Region Hannover")

# %% drop Klasse column
farmdata_df = df_farmI.drop(columns=['Klasse', 'LF_ha'])
farmdata_df['year'] = farmdata_df['year'].astype(int)

# rename columns in farmdata_df
farmdata_df = farmdata_df.rename(columns={
    'LF_km2': 'total_UAA_km2',
    'LF_mean': 'mean_UAA_km2',
    'Anzahl': 'farm_count'})

# save to csv
farmdata_df.to_csv(data_main_path+'/interim/farmdata.csv', index=False)

