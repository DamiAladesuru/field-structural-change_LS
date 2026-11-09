"""
Non-productive land category investigation
=============================================
Kept for reference / possible next revision round.

Categories under review:
    'Sonstige Flächen',
    'Stilllegung/Aufforstung',
    'Aus der Produktion/Erzeugung genommen',
    'AUKM',
    'GLK',
    'Sonstige Ökoregeln'

PS: zero Greening / Landschaftselement categories are present in the used cleaned dataset,
so landscape features (hedgerows, buffer strips) are not a concern here.
"""
import os
import pandas as pd
from pathlib import Path

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "workspace": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break

project_root=os.getcwd()

# set the path to the data folder
data_main_path=open(project_root+"/datapath.txt").read()

from src.data.dataprocessing_utils import field_dataset as fd

# %%
years = range(2012, 2025)
gld_no_geom = fd.process_nogeom_data(years=years, return_mode='without_small')

# %%
# ==========================================================
# 1. Count rows per Gruppe, per year
# ==========================================================
for year, df in gld_no_geom.items():
    print(f"\n=== {year} ===")
    print(df['Gruppe'].value_counts())


# %%
# ==========================================================
# 2. Build a year x Gruppe percentage table
# ==========================================================
rows = []

for year, df in gld_no_geom.items():
    counts = df['Gruppe'].value_counts()
    pct = (counts / len(df) * 100).round(2)
    for gruppe, value in pct.items():
        rows.append({'Gruppe': gruppe, 'year': year, 'pct': value})

long_df = pd.DataFrame(rows)

gruppe_table = long_df.pivot(index='Gruppe', columns='year', values='pct')
gruppe_table = gruppe_table.sort_index(axis=1)  # ensure years are in order

print(gruppe_table)

# Optionally save
gruppe_table.to_csv("reports/statistics/gldgruppe_percent_by_year.csv")


# %%
# ==========================================================
# 3. Sum proportions of the non-productive categories, per year
# ==========================================================
categories_to_sum = [
    'Sonstige Flächen',
    'Stilllegung/Aufforstung',
    'Aus der Produktion/Erzeugung genommen',
    'AUKM',
    'GLK',
    'Sonstige Ökoregeln'
]

# Check which of these actually exist in the table (avoids KeyError if a label differs slightly)
existing_categories = [c for c in categories_to_sum if c in gruppe_table.index]
missing_categories = [c for c in categories_to_sum if c not in gruppe_table.index]
if missing_categories:
    print(f"Warning: these categories were not found in the table and will be skipped: {missing_categories}")

summed_row = gruppe_table.loc[existing_categories].sum()
summed_row.name = 'Sum (non-production categories)'

gruppe_table_with_sum = pd.concat([gruppe_table, summed_row.to_frame().T])
print(gruppe_table_with_sum)

gruppe_table_with_sum.to_csv("reports/statistics/gldgruppe_percent_by_year_with_sum.csv")

# %%
for year, df in gld_no_geom.items():
    count = (df['Gruppe'] == 'Greening/Landschaftselement').sum()
    print(f"{year}: {count} rows")
    
# %%
# ==========================================================
# 4. Drop the non-productive categories from gld_no_geom
#    (NOT currently applied to the submitted version -- kept here
#     for the next revision round)
# ==========================================================
categories_to_drop = [
    'Sonstige Flächen',
    'Stilllegung/Aufforstung',
    'Aus der Produktion/Erzeugung genommen',
    'AUKM',
    'GLK',
    'Sonstige Ökoregeln'
]

gld_no_geom_excl_nonprod = {}
for year, df in gld_no_geom.items():
    before = len(df)
    gld_no_geom_excl_nonprod[year] = df[~df['Gruppe'].isin(categories_to_drop)]
    after = len(gld_no_geom_excl_nonprod[year])
    print(f"{year}: dropped {before - after} rows ({before} → {after})")

