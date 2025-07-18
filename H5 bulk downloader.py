### tetsing rubbish
import h5py
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from glob import glob
import re
import traceback
from tqdm import tqdm

# Folder containing H5 files
h5_folder = r"C:\Users\Filk\Desktop\Personal Research\NTLv4"
shapefile_path = r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\Shapefiles Dataset\ukr_admbnda_adm1_sspe_20240416.shp"
output_csv = r"C:\Users\Filk\Desktop\ntl_results.csv"

# Load shapefile for Ukraine oblasts
print("Loading shapefile...")
ukraine_oblasts = gpd.read_file(shapefile_path)
if ukraine_oblasts.crs != "EPSG:4326":
    ukraine_oblasts = ukraine_oblasts.to_crs(epsg=4326)
print(f"Loaded {len(ukraine_oblasts)} oblasts")

# Create Ukraine bounding box for preliminary filtering
ukraine_bbox = ukraine_oblasts.total_bounds  # [minx, miny, maxx, maxy]

# Get all H5 files
h5_files = glob(os.path.join(h5_folder, "*.h5"))
print(f"Found {len(h5_files)} H5 files to process")

# Function to find dataset paths
def find_datasets(h5_file):
    """Automatically discover dataset paths in HDF5 file"""
    results = {
        'ntl_path': None,
        'lat_path': None,
        'lon_path': None
    }
    
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Check for NTL data
            if 'Snow_Free' in name and 'NearNadir' in name:
                results['ntl_path'] = name
            # Check for latitude data
            elif 'Latitude' in name or 'lat' in name.lower():
                results['lat_path'] = name
            # Check for longitude data
            elif 'Longitude' in name or 'lon' in name.lower():
                results['lon_path'] = name
    
    with h5py.File(h5_file, 'r') as f:
        f.visititems(visitor)
    
    return results

# Process files
all_results = []
error_files = []

for file_path in tqdm(h5_files, desc="Processing files"):
    filename = os.path.basename(file_path)
    try:
        # Extract date from filename using regex
        date_match = re.search(r"\.A(\d{7})\.", filename)
        if date_match:
            date_str = date_match.group(1)
            year = int(date_str[:4])
            day_of_year = int(date_str[4:])
        else:
            # Try alternative naming pattern
            date_match = re.search(r"_(\d{4})(\d{3})_", filename)
            if date_match:
                year = int(date_match.group(1))
                day_of_year = int(date_match.group(2))
            else:
                year = 1900
                day_of_year = 1
                print(f"⚠️ Could not extract date from filename: {filename}")
        
        # Discover dataset paths
        dataset_paths = find_datasets(file_path)
        
        if not all(dataset_paths.values()):
            missing = [k for k, v in dataset_paths.items() if not v]
            print(f"❌ Missing datasets in {filename}: {', '.join(missing)}")
            error_files.append((filename, f"Missing datasets: {', '.join(missing)}"))
            continue
        
        # Load datasets
        with h5py.File(file_path, 'r') as f:
            ntl_data = f[dataset_paths['ntl_path']][:]
            lat = f[dataset_paths['lat_path']][:]
            lon = f[dataset_paths['lon_path']][:]
        
        # Handle different array dimensions
        if lat.ndim == 1 and lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        elif lat.ndim == 2 and lon.ndim == 2:
            lon_grid = lon
            lat_grid = lat
        else:
            raise ValueError(f"Unexpected coordinate dimensions: lat={lat.ndim}D, lon={lon.ndim}D")
        
        # Create points dataframe
        points = gpd.GeoDataFrame(
            {'ntl': ntl_data.ravel()},
            geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
            crs="EPSG:4326"
        )
        
        # Filter to Ukraine area
        points = points.cx[ukraine_bbox[0]:ukraine_bbox[2], ukraine_bbox[1]:ukraine_bbox[3]]
        
        if len(points) == 0:
            print(f"⚠️ No points within Ukraine bounding box in {filename}")
            error_files.append((filename, "No points in Ukraine bbox"))
            continue
        
        # Spatial join
        joined = gpd.sjoin(points, ukraine_oblasts, how='inner', predicate='within')
        
        if len(joined) == 0:
            print(f"⚠️ No points within Ukraine oblasts in {filename}")
            error_files.append((filename, "No points in oblasts"))
            continue
        
        # Aggregate results
        oblast_ntl = joined.groupby('ADM1_EN')['ntl'].mean().reset_index()
        oblast_ntl['year'] = year
        oblast_ntl['day_of_year'] = day_of_year
        oblast_ntl['filename'] = filename
        oblast_ntl['date'] = pd.to_datetime(f'{year}-{day_of_year}', format='%Y-%j')
        
        all_results.append(oblast_ntl)
        print(f"✅ Processed {filename}: {len(oblast_ntl)} oblasts")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Error processing {filename}: {error_msg}")
        error_files.append((filename, error_msg))
        traceback.print_exc()

# Combine results
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ Successfully processed {len(all_results)} files")
    print(f"📊 Results saved to: {output_csv}")
    
    # Add summary
    oblast_counts = final_df['ADM1_EN'].value_counts()
    date_range = f"{final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}"
    print(f"📅 Date range: {date_range}")
    print(f"📍 {len(oblast_counts)} oblasts covered")
    print(f"📈 Total records: {len(final_df)}")
else:
    print("\n❌ No files processed successfully")

# Report errors
if error_files:
    print("\n❌ Files with errors:")
    for filename, error in error_files:
        print(f"- {filename}: {error}")
    print(f"Total errors: {len(error_files)}")


