---
contributors:
  - Damilola Aladesuru
  - Josef Baumert

---

## Overview

This repository contains the details for replicating the paper "Spatio-temporal Patterns of Field Structural Change and Related Farm Developments in Lower Saxony" by Damilola Aladesuru, Josef Baumert, Christine Wieck, Till Kuhn, Hugo Storm.

### Memory and Runtime Requirements
All processes in this repository have been run using a CPU. 

### Description of programs/code

Organization of the code largely follows the Cookiecutter Data Science template (https://drivendata.github.io/cookiecutter-data-science/)

The script-containing parts are:
- Programs in `src/data` are used to prepare and process data sets. 
- Programs in `src/analyses_results` are used to analysed processed data. 

#### Processing the field data
`src/data/dataprocessing_utils/gridregionjoin.py` joins the eea grid with the district boundaries of Niedersachsen.
Output: data/interim/grid_landkreise.pkl - a GeoDataFrame with the grid cells that intersect with the landkreise boundaries. The geodataframe is called and used in the dataload script to assign district and grid cell to the field polygons.

`src/data/dataprocessing_utils/dataload.py` is the primary data loading script that loads the raw data, drops unnecessary columns, removes areas outside of Niedersachsen boundaries, spatially joins all years with district information (Kreise) and the EEA reference 10km grid, and prepares data for further processing and analysis.
Output: parquet files for each year – data/interim/gld_per_year

After data loading, we use `src/data/dataprocessing_utils/eca_new.py` to refactor and harmonise land-use information across all years in the study data, producing a land-use master file - data/interim/kulturcode_mastermap.csv. This master file is applied in `src/data/processing_data_utils/field_dataset.py` to identify land-use group values across years in the full dataset.

`src/data/dataprocessing_utils/field_dataset.py` is the final field-level processing script before aggregations to grid cells and districts. In addition to updating land-use information in the raw data using the created master map, the script filters out polygons from unproductive groups and excludes polygons with an area of less than 100 m².

#### Deriving coarse-level field aggregates and computing change
`src/data/coarseaggregates_andchange_utils/griddf_desc.py` contains code for obtaining grid-level field aggregates and computing change at this level while `src/data/coarseaggregates_andchange_utils/landkreis_desc.py` does the same for the district level.

#### Processing the farm data
`src/data/dataprocessing_utils/process_farmdata.py` loads farm data xml files, drops unneeded columns such as Klasse, tidies the district id, convert ha columns to km² and saves as csv
Output: data/interim/farmdata.csv

#### Answering RQs 1 and 2
`src/analyses_results/RQ1_line_kde.py` provide the plot and values in the line plot and kde section of 3.1, `src/analyses_results/RQ1_spatial.py` provide Figure 5 and stats, and `src/analyses_results/RQ2_analysis_andplot.py` computes change infarm for 2016 and 2020, extracts needed field change and provides stats and figures for 3.2

**Data checks**
We use the scripts in `src/data/datachecks` to assess and understand the field data, e.g., checking differences between schlag and teil schlag, identifying non-productive polygon groups in the data, deriving the statistics provided in the methodology, and detecting examples of field structural change. `nonproductivecheck.py` and `stat_in_methodology.py` use the full dataset while others use only some parts of the data. Since these parts were identified manually, using these other scripts exactly as is may raise errors if the data is customised differently, but the scripts still give an idea of the checks we carried out.


### License for Code

The code is licensed under a Apache License 2.0.


## Project Setup
Ensure devcontainer setup works for you.

### Requirements

- **Python:** 3.x
- **Packages:** Full list of packages to install and their versions are provided in the `requirements.txt` file.

### Data
The complete yearly data for all 13 years in the study are confidential and must be requested from the Lower Saxony Center for Rural Development and Agricultural Promotion (Servicezentrum Landentwicklung und Agrarförderung). However, the data for a current and previous year (e.g., current: 2026 and previous: 2025) are public accessible via https://sla.niedersachsen.de/mapbender_sla/application/Schlaginfo.


This table provides information about the additional datasets used along with the field data for this study.
| Data.Name  | Needed.Data.Files | Storage.Location | Publicly.Accessible | Link|
| -- | -- | -- | -- |-- |
| "State boundaries" | `NDS_Landesflaeche.shp`; `NDS_Landkreise.shp` | `.../verwaltungseinheiten/` | TRUE | [OpenGeoData](https://ni-lgln-opengeodata.hub.arcgis.com/documents/e9631a1bccac4823912963aa8f0bc0b0/about)|
| "10km grid" |  `eea_10_km_eea-ref-grid-de_p_2013_v02_r00/de_10km.shp` |  `data/raw/` | TRUE | [EEA Datahub](https://www.eea.europa.eu/en/datahub/datahubitem-view/3c362237-daa4-45e2-8c16-aaadfb1a003b) |
| "Farm" |  `farmdata.zip` |  `data/raw/` | TRUE | [Regional database](https://www1.nls.niedersachsen.de/statistik/html/default.asp) |
---

**Data access guide**

***State boundaries***:
1. Visit the provided link
2. Click `Read More` under Summary
3. Click the shapefile link under Download
You can save this downloaded zipped file anywhere you want. Just make sure the unzipped folder is saved in `data/raw`. From the files contained in the folder, you will need `NDS_Landesflaeche.shp` for the state’s boundary and `NDS_Landkreise.shp` for the NUTS3 regions of Lower Saxony.

***10km grid***:
1. Visit the provided link
2. Click `2013` under Datasets
3. In the `Type to filter` box, type `Germany`
4. Click the 10km dropdown and you will find a direct download link. Click this
5. On the page to which you are redirected, click `Download all files`
You can save this downloaded zipped file anywhere you want. Just make sure the unzipped folder is saved in `data/raw`.

***Farm data***:
1. Click on the farm data link
2. Click `WEITER` -> `4 - Wirtschaftsbereiche, Verkehr` -> `41 - Land- und Forstwirtschaft, Fischerei` -> `41121 Agrarstrukturerhebung` -> `Landwirtschaftliche Betriebe mit LF nach Größenklasse der LF (2010,2016,2020)`
3. Click `Kreisfr. Stadt, Landkreis` -> `alle auswählen` -> OK
4. Unselect 0, 1, 2, 3, 4
5. Click `Table erstellen` -> `gezippte Excel-Datei herunterladen`
6. Rename downloaded zipped file and move to `data/raw`

---

---
## Replication Workflow
1. Run `src/data/dataprocessing_utils/process_farmdata.py` to process farm data. This step should take only a couple seconds.

2. Run `src/data/datachecks/stats_in_methodology.py`. This will take several minutes but at completion, all the steps for processing field data and coarse-level aggregation would have been completed. Also, the statistics mentioned in methodology section would have been derived.
Line **43* which does `allgld_no_geom = fd.process_nogeom_data(years=years, return_mode='all')` is a heavy step that processes field data. Thus, due to the large data, it is possible that you will experience kernel crashes while at this stage. If this happens, restart editor and start from the needed chunk.  The field data processing scripts contain programs that checks if a year has been processed thus preventing always rerunning the process for all years.

3. Run `src/analyses_results` scripts to produce the results section

Use other files in `src/data/datachecks` as needed.

---

---


## Acknowledgements


This readme was created following the guidelines from [Hindawi](https://social-science-data-editors.github.io/template_README/).
