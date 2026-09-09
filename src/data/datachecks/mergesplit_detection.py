# %%
import geopandas as gpd

# this code doesn't detect perfectly. i later modified some of tghe detected changes manually in arcgs.
# so the process is mostly fully arcgis.

union_gdf = gpd.read_file("/workspace/data/interim/fahga/fahga.shp")  # file containing union of 2020 and 2023 sample fields
print(union_gdf.columns)

# %%
flik_2020_col = "FLIK_overl"       # replace with your actual 2020 FLIK column name
schlag_2020_col = "SCHLAGNR_o" # replace with your actual 2020 SCHLAG_NR column name
flik_2023_col = "FLIK"         # replace with your actual 2023 FLIK column name
schlag_2023_col = "SCHLAGNR"   # replace with your actual 2023 SCHLAG_NR column name

# %%
# Build combined FLIK+SCHLAG_NR keys for each year (avoids false matches from
# SCHLAG_NR alone repeating across different FLIKs)
union_gdf['id_2020'] = union_gdf[flik_2020_col].astype(str) + "_" + union_gdf[schlag_2020_col].astype(str)
union_gdf['id_2023'] = union_gdf[flik_2023_col].astype(str) + "_" + union_gdf[schlag_2023_col].astype(str)

# A row only "has" a 2020/2023 id if the underlying FLIK+SCHLAG_NR fields are both present
union_gdf['has_2020'] = union_gdf[flik_2020_col].notna() & union_gdf[schlag_2020_col].notna()
union_gdf['has_2023'] = union_gdf[flik_2023_col].notna() & union_gdf[schlag_2023_col].notna()

# %%
# Disappeared: has 2020 id, no 2023 id
disappeared = union_gdf[union_gdf['has_2020'] & ~union_gdf['has_2023']]

# Appeared: has 2023 id, no 2020 id
appeared = union_gdf[union_gdf['has_2023'] & ~union_gdf['has_2020']]

# For split/merge, look at rows where BOTH ids exist
both = union_gdf[union_gdf['has_2020'] & union_gdf['has_2023']]

# Count how many distinct 2023 ids each 2020 id maps to (split if >1)
split_counts = both.groupby('id_2020')['id_2023'].nunique()
split_ids = split_counts[split_counts > 1].index
split = both[both['id_2020'].isin(split_ids)]

# Count how many distinct 2020 ids each 2023 id maps to (merged if >1)
merge_counts = both.groupby('id_2023')['id_2020'].nunique()
merge_ids = merge_counts[merge_counts > 1].index
merged = both[both['id_2023'].isin(merge_ids)]

print(f"Disappeared: {disappeared['id_2020'].nunique()}")
print(f"Appeared: {appeared['id_2023'].nunique()}")
print(f"Split: {len(split_ids)}")
print(f"Merged: {len(merge_ids)}")

# %%
union_gdf['change_type'] = 'unchanged'
union_gdf.loc[union_gdf['id_2020'].isin(disappeared['id_2020']), 'change_type'] = 'disappeared'
union_gdf.loc[union_gdf['id_2023'].isin(appeared['id_2023']), 'change_type'] = 'appeared'
union_gdf.loc[union_gdf['id_2020'].isin(split_ids), 'change_type'] = 'split'
union_gdf.loc[union_gdf['id_2023'].isin(merge_ids), 'change_type'] = 'merged'

# %%
objectids_to_reset = [1, 2, 3]  # replace with your actual list of OBJECTID values

union_gdf.loc[union_gdf['OBJECTID'].isin(objectids_to_reset), 'change_type'] = 'unchanged'
union_gdf.to_file("/workspace/data/interim/fahga/union_classified.shp")
# %%