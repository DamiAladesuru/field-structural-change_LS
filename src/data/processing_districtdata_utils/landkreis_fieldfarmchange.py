# %% Import necessary libraries
import zipfile
import pandas as pd
from pathlib import Path
import os, re
import xml.etree.ElementTree as ET
import numpy as np

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

###########################################
# %% load needed existing data
landkreisdf = pd.read_parquet(os.path.join(
    data_main_path, "interim", "landkreisdf", "landkreisdf_crop_elev.parquet"
))

#############################################################
# import farm data
#############################################################
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

#compute LF_mean
df_farmI['LF_mean'] = df_farmI['LF_ha'] / df_farmI['Anzahl']

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
farmdata_ge = df_farmI.drop(columns=['Klasse'])
 
# %%
# Get unique LANDKREIS values
farm_landkreise = set(farmdata_ge["LANDKREIS"].unique())

print(farmdata_ge['year'].unique())

farmdata_ge['year'] = farmdata_ge['year'].astype(int)

# %%
print(farmdata_ge.info())

# %% rename columns in farmdata_ge
farmdata_ge = farmdata_ge.rename(columns={
    'LF_ha': 'total_UAA',
    'LF_mean': 'mean_UAA',
    'Anzahl': 'farm_count'})

# save farmdata_ge to csv
farmdata_ge.to_csv(data_main_path+'/interim/landkreis_field_farm_elev_maincrop.csv', index=False)


#####################################################################
# Compute farm value differences
#####################################################################
#%% Load the cleaned DataFrame from CSV
farmdata_ge = pd.read_csv(data_main_path+'/interim/landkreis_field_farm_elev_maincrop.csv')

farmdata_ge_ext = farmdata_ge.copy()
# drop 2010
farmdata_ge_ext = farmdata_ge_ext[farmdata_ge_ext['year']!= 2010]
# reset index to avoid issues with groupby
farmdata_ge_ext.reset_index(drop=True, inplace=True)

#resort
farmdata_ge_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
numeric_columns = farmdata_ge_ext.select_dtypes(include='number').columns

# Create a dictionary to store the new columns
new_columns = {}

# Loop through each numeric column
for col in numeric_columns:
    # Difference from first year value within each group
    new_columns[f'{col}_diff_to_y2016'] = (
        farmdata_ge_ext[col] - farmdata_ge_ext.groupby('LANDKREIS')[col].transform('first')
    )

    # Percentage difference from first year value within each group
    new_columns[f'{col}_percdiff_to_y2016'] = (
        (farmdata_ge_ext[col] - farmdata_ge_ext.groupby('LANDKREIS')[col].transform('first')) / 
        farmdata_ge_ext.groupby('LANDKREIS')[col].transform('first')
    ).fillna(0) * 100


# Concatenate the new columns to the original DataFrame all at once
new_columns_df = pd.DataFrame(new_columns)
farmdata_ge_ydiff = pd.concat([farmdata_ge_ext, new_columns_df], axis=1)
farmdata_ge_ydiff.head()

# %%
farmdata_ge_ydiff = farmdata_ge_ydiff[farmdata_ge_ydiff['year']== 2020]

# %%
columns_to_keep = [
    'LANDKREIS', 'year',
    'areakm2_sum', 'fields', 'medshape',
    'medareakm2_percdiff_to_y2016',
    'areakm2_sum_percdiff_to_y2016',
    'fields_percdiff_to_y2016',
    'fields_perarea_percdiff_to_y2016',
    'medshape_percdiff_to_y2016',
    'main_crop', 'main_crop_group'
]

# Drop all other columns from landkreis level field change data
df_landks = landkreisdf[columns_to_keep]

# %% merge
field_farm = farmdata_ge_ydiff.merge(df_landks, on=["LANDKREIS", "year"])

field_farm.info()
field_farm.to_parquet(data_main_path+"/interim/landkreisdf/field_farmchange.parquet")

# %% create farmcount map df
landkreis_farmdesc = farmdata_ge[['LANDKREIS', 'year', 'farm_count', 'total_UAA']]
landkreis_farmdesc = landkreis_farmdesc[( landkreis_farmdesc['year'] == 2016)]

farmcountdf = landkreis_farmdesc.merge(farmdata_ge_ydiff[['LANDKREIS', 'farm_count_percdiff_to_y2016']], on=["LANDKREIS"])
farmcountdf.to_parquet(data_main_path+"/interim/landkreisdf/farmcount_change.parquet")
# %%