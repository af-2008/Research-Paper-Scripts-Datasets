This study investigates the extent to which international trade integration reduces the economic impact of armed conflict at the subnational level, using panel data from 27 Ukrainian oblasts between 2012 and 2022. Night-time light (NTL) intensity serves as a proxy for economic activity, and conflict severity is drawn from geocoded event-level data
 Contents:

All datasets are either publicly available or derived from publicly available sources:

Night-Time Light Data (2012–2022)
Source: VIIRS (NOAA)
Format: .h5 files (aggregated to oblast-year level)

Conflict Exposure (2012–2022)
Source: ACLED Ukraine Dataset
Metric: ln(1 + events + fatalities) by region and year

Trade Integration (2012–2022)
Source: Ukraine customs and regional trade reports (from .xls and .mhtml files)
Metric: (Imports + Exports) / Regional GDP

Population by Oblast
Source: Ukraine State Statistics Service (2012–2022)

World Development Indicators
Source: World Bank Open Data
Used for robustness controls (urbanisation, density)

Shapefiles
Source: Humanitarian Data Exchange – Ukraine Admin Boundaries: https://data.humdata.org/

SCRIPTS:

**Data cleaning conflict exposure:**           Script that takes raw data and helps clean it and remove unnecessary info about conflict.

**Merging Datasets V2**                        Script that merges all the datasets and csv files 

**Data Cleaning script for trade:**            Script that cleans a raw dataset on trade

**Figure Visualiser:**                         Script that helps build figures and tables and plots 

**H5 bulk downloader:**                        Script that takes the NTL data and helps me automatically download it all





