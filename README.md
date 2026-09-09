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
- Programs in `src/analysis` are used to analysed processed data.
- Programs in `src/visualization` are used to visualize and to prepare the results. 

### License for Code

The code is licensed under a Apache License 2.0.



## Project Setup
Ensure devcontainer setup works for you.

### Requirements

- **Python:** 3.x
- **Packages:** Full list of packages to install and their versions are provided in the `requirements.txt` file.

### Data
The links and download procedure for the field data are embedded in the data-loading workflow of script `src/data/processing_fielddata_utils/dataload.py`. To download the data outside of this replication workflow, copy out the data urls and use as preferred.

This table provides information about the supplementary datasets used along with the field data for this study.
| Data.Name  | Needed.Data.Files | Storage.Location | Publicly.Accessible | Link|
| -- | -- | -- | -- |-- |
| "State boundaries" | `NDS_Landesflaeche.shp`; `NDS_Landkreise.shp` | `.../verwaltungseinheiten/` | TRUE | [OpenGeoData](https://ni-lgln-opengeodata.hub.arcgis.com/documents/e9631a1bccac4823912963aa8f0bc0b0/about)|
| "10km grid" |  `eea_10_km_eea-ref-grid-de_p_2013_v02_r00/de_10km.shp` |  `data/raw/` | TRUE | [EEA Datahub](https://www.eea.europa.eu/en/datahub/datahubitem-view/3c362237-daa4-45e2-8c16-aaadfb1a003b) |
| "Elevation" |  `lsax_elevation.tif` |  `data/raw/` | TRUE | [Google Earth Engine](https://code.earthengine.google.com/dfce28651cb69cbbdfd1e71134294643) |
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

***Elevation data***:
To access Google Earth Engine, you need a Google account
 - Go to https://earthengine.google.com/platform/ and click Get Started
 - On the displayed configuration page, choose applicable use category. If non-commercial as in the case of this study, click Get Started under `See if you are eligible for non-commercial use`. This page will lead you to a registration page where you can provide work and organisation information. After completing the steps, you should be registered for non-commercial GEE use.

Once the registration is completed:
1. Click the GEE link in the data table above
2. Click on Tasks in the top right panel
3. Run `lsax_elevation`
4. Change folder name to an existing Google Drive folder you can later access or create folder `elevation_fisc` in your Drive
5. Click `Run`
Download lsax_elevation.geo_tiff file from Google drive and place in `data/raw`


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
1. Run `src/data/datachecks/data_exploration.py` in chunks as sectioned with '# %%'.
  This processes the data while utilizing the helper scripts in `src/data/processing_fielddata_utils` and `src/data/processing_griddata_utils`. It also allows exploring the data, e.g., obtaining the information provided in Appendix A, and verifying the descriptive values and shape figures provided in the methods section.

  Due to the large data, it is possible that you will experience kernel crashes while running the code chunks e.g., after running the first three chunks (excluding project setup). If this happen, restart editor and continue from where you stopped except if of course you need the results from a previous step. Then start from the needed chunk.

2. Run `src/visualization/lineplots.py` to generate Figure 4 and 5

3. Run `src/visualization/mapplots.py` to generate Figure 6 and descriptive statistics of the plotted maps

4. Process farm data and merge to district data by running files in `src/data/processing_districtdata_utils` starting with landkreis_elev and then landkreis_fieldfarmchange.py`

5. Run `src/analysis/district_farmfield_relationship.py` and then `src/analysis/district_farmcountchange.py`. These generate the maps and scatterplots in Figure 7 and 8, the kde in Figure B1, maps in Figure B2 and the correlation coefficients in Table B3.

---

---


## Acknowledgements


This readme was created following the guidelines from [Hindawi](https://social-science-data-editors.github.io/template_README/).
