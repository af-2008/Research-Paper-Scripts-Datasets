import os
import re
import glob
import h5py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = r"C:\Users\Filk\Desktop\Personal Research"
TRADE_CSV     = os.path.join(DATA_DIR, "Clean data", "ukraine_trade_data_clean_translated.csv")
NTL_DIR       = os.path.join(DATA_DIR, "NTL_HDF5")
CONFLICT_CSV  = os.path.join(DATA_DIR, "Clean data", "conflict_events_clean.csv")
POP_CSV       = os.path.join(DATA_DIR, "population_by_oblast.csv")
WORLD_IND_CSV = os.path.join(DATA_DIR, "world_indicators.csv")
SHAPEFILE     = os.path.join(DATA_DIR, "ukr_admbnda_adm1_sspe_20240416.shp")
YEARS         = list(range(2012, 2023))

# ─── LOAD OBLASTS ──────────────────────────────────────────────────────────────
def load_oblasts(shapefile=SHAPEFILE):
    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    gdf['region'] = gdf['ADM1_EN'].str.lower().str.strip()
    return gdf[['region','geometry']]

# ─── TRADE ─────────────────────────────────────────────────────────────────────
def aggregate_trade():
    df = pd.read_csv(TRADE_CSV, encoding='utf-8-sig')
    df['region'] = df['region'].str.lower().str.strip()
    return df

# ─── CONFLICT (SPATIAL JOIN) ───────────────────────────────────────────────────
def aggregate_conflict(oblasts_gdf):
    df = pd.read_csv(CONFLICT_CSV, parse_dates=['date_start'])
    # build geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    # spatial join
    joined = gpd.sjoin(gdf, oblasts_gdf, how='left', predicate='within')
    joined['year'] = joined['date_start'].dt.year
    # aggregate fatalities
    agg = joined.groupby(['region','year'])['best_est'].sum().reset_index()
    agg.rename(columns={'best_est':'conflict_fatalities'}, inplace=True)
    return agg

# ─── POPULATION ────────────────────────────────────────────────────────────────
def aggregate_population():
    pop = pd.read_csv(POP_CSV)
    pop.columns = pop.columns.str.lower().str.strip()
    pop['region'] = pop['region'].str.lower().str.strip()
    return pop[['region','year','population']]

# ─── WORLD INDICATORS ──────────────────────────────────────────────────────────
def aggregate_indicators():
    df = pd.read_csv(WORLD_IND_CSV)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ','_')
    df = df[df['year'].between(2012,2022)]
    df['region'] = df['region'].str.lower().str.strip()
    keep = ['region','year',
            'exports_of_goods_and_services_%_of_gdp',
            'imports_of_goods_and_services_%_of_gdp',
            'gdp_current_us$', 'gdp_growth_annual_%']
    return df[keep]

# ─── NTL (SPATIAL JOIN) ─────────────────────────────────────────────────────────
def aggregate_ntl(oblasts_gdf):
    rows = []
    for path in glob.glob(os.path.join(NTL_DIR,"*.h5")):
        year = int(re.search(r'(\d{4})', os.path.basename(path)).group(1))
        with h5py.File(path,'r') as f:
            lats = f['latitude'][:].ravel()
            lons = f['longitude'][:].ravel()
            vals = f['radiance'][:].ravel()   # adapt if dataset name differs
        df = pd.DataFrame({'lat':lats,'lon':lons,'val':vals})
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,df.lat), crs='EPSG:4326')
        joined = gpd.sjoin(gdf, oblasts_gdf, how='inner', predicate='within')
        mean_vals = joined.groupby('region')['val'].mean().reset_index()
        mean_vals['year'] = year
        rows.append(mean_vals)
    ntl_df = pd.concat(rows, ignore_index=True)
    ntl_df.rename(columns={'val':'mean_ntl'}, inplace=True)
    return ntl_df

# ─── BUILD PANEL ───────────────────────────────────────────────────────────────
def build_panel():
    oblasts = load_oblasts()
    trade    = aggregate_trade()
    conflict = aggregate_conflict(oblasts)
    pop      = aggregate_population()
    indic    = aggregate_indicators()
    ntl      = aggregate_ntl(oblasts)

    # base grid region×year
    panel = (pd.MultiIndex.from_product([oblasts.region.unique(), YEARS],
                                        names=['region','year'])
             .to_frame(index=False))

    for df in [trade, conflict, pop, indic, ntl]:
        panel = panel.merge(df, on=['region','year'], how='left')

    # fill zeros
    panel['conflict_fatalities'] = panel['conflict_fatalities'].fillna(0)
    panel['mean_ntl'] = panel['mean_ntl'].fillna(0)
    return panel

if __name__=="__main__":
    panel_df = build_panel()
    panel_df.to_csv(os.path.join(DATA_DIR, "panel_data_2012_2022.csv"), index=False)
    print("Saved panel_data_2012_2022.csv, shape:", panel_df.shape)

