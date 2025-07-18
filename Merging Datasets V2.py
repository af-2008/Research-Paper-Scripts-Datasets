import pandas as pd
import numpy as np
import os
from glob import glob
import re

# ======================
# 1. STANDARDIZE REGIONS (COMPREHENSIVE MAPPING)
# ======================
# Master oblast name mapping dictionary with all variations
oblast_mapping = {
    # Kyiv and variations
    'kyiv': 'Kyiv',
    'kyiv city': 'Kyiv',
    'kyivska': 'Kyiv Oblast',
    'kyiv oblast': 'Kyiv Oblast',
    "kyivs'ka": 'Kyiv Oblast',
    'kyivska oblast': 'Kyiv Oblast',
    'kyiv region': 'Kyiv Oblast',
    'kyivcity': 'Kyiv',
    'kyivs ka': 'Kyiv Oblast',
    'kyiv special republican city': 'Kyiv',
    'kyivs ka oblast': 'Kyiv Oblast',
    'm kyiv': 'Kyiv',
    'city of kyiv': 'Kyiv',
    'mkiyiv': 'Kyiv',
    'kiev': 'Kyiv',
    'kiev city': 'Kyiv',
    'kiev oblast': 'Kyiv Oblast',
    'kiev special republican city': 'Kyiv',
    'kiev municipality': 'Kyiv',
    'kyiv municipality': 'Kyiv',
    'kyivcity': 'Kyiv',
    'm kiyiv': 'Kyiv',
    
    # Crimea and variations
    'crimea': 'Crimea',
    'crimea autonomous republic': 'Crimea',
    'autonomous republic of crimea': 'Crimea',
    'ark': 'Crimea',
    'crimea republic': 'Crimea',
    
    # Sevastopol and variations
    'sevastopol': 'Sevastopol',
    'sevastopol city': 'Sevastopol',
    'sevastopol city state administration': 'Sevastopol',
    'm sevastopol': 'Sevastopol',
    'city of sevastopol': 'Sevastopol',
    
    # Cherkasy Oblast
    'cherkasy': 'Cherkasy Oblast',
    'cherkasy oblast': 'Cherkasy Oblast',
    'cherkaska': 'Cherkasy Oblast',
    'cherkas ka': 'Cherkasy Oblast',
    'cherkaska oblast': 'Cherkasy Oblast',
    
    # Chernihiv Oblast
    'chernihiv': 'Chernihiv Oblast',
    'chernihiv oblast': 'Chernihiv Oblast',
    'chernihivska': 'Chernihiv Oblast',
    'chernigivska': 'Chernihiv Oblast',
    'chernihivs ka': 'Chernihiv Oblast',
    
    # Chernivtsi Oblast
    'chernivtsi': 'Chernivtsi Oblast',
    'chernivtsi oblast': 'Chernivtsi Oblast',
    'chernivetska': 'Chernivtsi Oblast',
    'chernivtsi': 'Chernivtsi Oblast',
    'chernivetska oblast': 'Chernivtsi Oblast',
    
    # Dnipropetrovsk Oblast
    'dnipropetrovsk': 'Dnipropetrovsk Oblast',
    'dnipropetrovsk oblast': 'Dnipropetrovsk Oblast',
    'dnipro': 'Dnipropetrovsk Oblast',
    'dnipropetrovska': 'Dnipropetrovsk Oblast',
    'dnipropetrovs ka': 'Dnipropetrovsk Oblast',
    'dnipropetrovska oblast': 'Dnipropetrovsk Oblast',
    
    # Donetsk Oblast
    'donetsk': 'Donetsk Oblast',
    'donetsk oblast': 'Donetsk Oblast',
    'donetska': 'Donetsk Oblast',
    'donets ka': 'Donetsk Oblast',
    'donetska oblast': 'Donetsk Oblast',
    
    # Ivano-Frankivsk Oblast
    'ivano-frankivsk': 'Ivano-Frankivsk Oblast',
    'ivano-frankivsk oblast': 'Ivano-Frankivsk Oblast',
    'ivanofrankivsk': 'Ivano-Frankivsk Oblast',
    'ivano frankivsk': 'Ivano-Frankivsk Oblast',
    'ivanofrankivska': 'Ivano-Frankivsk Oblast',
    'ivano-frankivska': 'Ivano-Frankivsk Oblast',
    'ivano96frankivsk': 'Ivano-Frankivsk Oblast',
    'ivano-frankivs ka': 'Ivano-Frankivsk Oblast',
    
    # Kharkiv Oblast
    'kharkiv': 'Kharkiv Oblast',
    'kharkiv oblast': 'Kharkiv Oblast',
    'kharkivska': 'Kharkiv Oblast',
    'kharkivs ka': 'Kharkiv Oblast',
    'kharkivska oblast': 'Kharkiv Oblast',
    
    # Kherson Oblast
    'kherson': 'Kherson Oblast',
    'kherson oblast': 'Kherson Oblast',
    'khersonska': 'Kherson Oblast',
    'khersons ka': 'Kherson Oblast',
    'khersonska oblast': 'Kherson Oblast',
    
    # Khmelnytskyi Oblast
    'khmelnytskyi': 'Khmelnytskyi Oblast',
    'khmelnytskyi oblast': 'Khmelnytskyi Oblast',
    'khmelnytsky': 'Khmelnytskyi Oblast',
    'khmelnytskiy': 'Khmelnytskyi Oblast',
    'khmelnytska': 'Khmelnytskyi Oblast',
    'khmelnytska oblast': 'Khmelnytskyi Oblast',
    'khmelnitska': 'Khmelnytskyi Oblast',
    'khmelnitska oblast': 'Khmelnytskyi Oblast',
    'khmelnytska': 'Khmelnytskyi Oblast',
    
    # Kirovohrad Oblast
    'kirovohrad': 'Kirovohrad Oblast',
    'kirovohrad oblast': 'Kirovohrad Oblast',
    'kirovohradska': 'Kirovohrad Oblast',
    'kirovohrads ka': 'Kirovohrad Oblast',
    'kirovohradska oblast': 'Kirovohrad Oblast',
    'kirovogradska': 'Kirovohrad Oblast',
    'kirovogradska oblast': 'Kirovohrad Oblast',
    
    # Luhansk Oblast
    'luhansk': 'Luhansk Oblast',
    'luhansk oblast': 'Luhansk Oblast',
    'luhanska': 'Luhansk Oblast',
    'luhanska oblast': 'Luhansk Oblast',
    'luganska': 'Luhansk Oblast',
    'luganska oblast': 'Luhansk Oblast',
    
    # Lviv Oblast
    'lviv': 'Lviv Oblast',
    'lviv oblast': 'Lviv Oblast',
    'lvivska': 'Lviv Oblast',
    'lvivs ka': 'Lviv Oblast',
    'lvivska oblast': 'Lviv Oblast',
    
    # Mykolaiv Oblast
    'mykolaiv': 'Mykolaiv Oblast',
    'mykolaiv oblast': 'Mykolaiv Oblast',
    'mykolayiv': 'Mykolaiv Oblast',
    'mykolaivska': 'Mykolaiv Oblast',
    'mykolayivska': 'Mykolaiv Oblast',
    'mikolayivska': 'Mykolaiv Oblast',
    'mykolaivs ka': 'Mykolaiv Oblast',
    
    # Odesa Oblast
    'odesa': 'Odesa Oblast',
    'odesa oblast': 'Odesa Oblast',
    'odessa': 'Odesa Oblast',
    'odeska': 'Odesa Oblast',
    'odes ka': 'Odesa Oblast',
    'odeska oblast': 'Odesa Oblast',
    
    # Poltava Oblast
    'poltava': 'Poltava Oblast',
    'poltava oblast': 'Poltava Oblast',
    'poltavska': 'Poltava Oblast',
    'poltavs ka': 'Poltava Oblast',
    'poltavska oblast': 'Poltava Oblast',
    
    # Rivne Oblast
    'rivne': 'Rivne Oblast',
    'rivne oblast': 'Rivne Oblast',
    'rivnenska': 'Rivne Oblast',
    'rivnens ka': 'Rivne Oblast',
    'rivnenska oblast': 'Rivne Oblast',
    
    # Sumy Oblast
    'sumy': 'Sumy Oblast',
    'sumy oblast': 'Sumy Oblast',
    'sumska': 'Sumy Oblast',
    'sums ka': 'Sumy Oblast',
    'sumska oblast': 'Sumy Oblast',
    
    # Ternopil Oblast
    'ternopil': 'Ternopil Oblast',
    'ternopil oblast': 'Ternopil Oblast',
    'ternopilska': 'Ternopil Oblast',
    'ternopils ka': 'Ternopil Oblast',
    'ternopilska oblast': 'Ternopil Oblast',
    
    # Vinnytsia Oblast
    'vinnytsia': 'Vinnytsia Oblast',
    'vinnytsia oblast': 'Vinnytsia Oblast',
    'vinnytsya': 'Vinnytsia Oblast',
    'vinnytska': 'Vinnytsia Oblast',
    'vinnitska': 'Vinnytsia Oblast',
    'vinnytska oblast': 'Vinnytsia Oblast',
    'vinnitska oblast': 'Vinnytsia Oblast',
    
    # Volyn Oblast
    'volyn': 'Volyn Oblast',
    'volyn oblast': 'Volyn Oblast',
    'volynska': 'Volyn Oblast',
    'volyns ka': 'Volyn Oblast',
    'volynska oblast': 'Volyn Oblast',
    'volinska': 'Volyn Oblast',
    
    # Zakarpattia Oblast
    'zakarpattia': 'Zakarpattia Oblast',
    'zakarpattia oblast': 'Zakarpattia Oblast',
    'zakarpatska': 'Zakarpattia Oblast',
    'zakarpats ka': 'Zakarpattia Oblast',
    'zakarpatska oblast': 'Zakarpattia Oblast',
    'zakarpattya': 'Zakarpattia Oblast',
    
    # Zaporizhzhia Oblast
    'zaporizhzhia': 'Zaporizhzhia Oblast',
    'zaporizhzhia oblast': 'Zaporizhzhia Oblast',
    'zaporizka': 'Zaporizhzhia Oblast',
    'zaporizhzhya': 'Zaporizhzhia Oblast',
    'zaporizka oblast': 'Zaporizhzhia Oblast',
    'zaporizhzhya oblast': 'Zaporizhzhia Oblast',
    'zaporiz zka': 'Zaporizhzhia Oblast',
    
    # Zhytomyr Oblast
    'zhytomyr': 'Zhytomyr Oblast',
    'zhytomyr oblast': 'Zhytomyr Oblast',
    'zhytomyrska': 'Zhytomyr Oblast',
    'zhitomirska': 'Zhytomyr Oblast',
    'zhytomyrs ka': 'Zhytomyr Oblast',
    'zhytomyrska oblast': 'Zhytomyr Oblast',
    
    # Common abbreviations and misspellings
    'dnipro': 'Dnipropetrovsk Oblast',
    'ivano': 'Ivano-Frankivsk Oblast',
    'khmelnytsky': 'Khmelnytskyi Oblast',
    'mykolayiv': 'Mykolaiv Oblast',
    'ternopil': 'Ternopil Oblast',
    'volyn': 'Volyn Oblast',
    'zakarpattia': 'Zakarpattia Oblast',
    'zaporizhzhia': 'Zaporizhzhia Oblast',
    'zhytomyr': 'Zhytomyr Oblast',
    'cherkasy': 'Cherkasy Oblast',
    'chernihiv': 'Chernihiv Oblast',
    'chernivtsi': 'Chernivtsi Oblast',
    'kherson': 'Kherson Oblast',
    'kirovohrad': 'Kirovohrad Oblast',
    'luhansk': 'Luhansk Oblast',
    'poltava': 'Poltava Oblast',
    'rivne': 'Rivne Oblast',
    'sumy': 'Sumy Oblast',
    'vinnytsia': 'Vinnytsia Oblast',
    
    # Special cases
    'vinnitska': 'Vinnytsia Oblast',
    'volinska': 'Volyn Oblast',
    'zhitomirska': 'Zhytomyr Oblast',
    'kiyivska': 'Kyiv Oblast',
    'luganska': 'Luhansk Oblast',
    'mikolayivska': 'Mykolaiv Oblast',
    'khmelnitska': 'Khmelnytskyi Oblast',
    'chernigivska': 'Chernihiv Oblast',
}
# ADDITIONAL MAPPINGS FOR REMAINING ISSUES
oblast_mapping.update({
    # Handle "Ukraine" (country-level data)
    'ukraine': None,  # Will be filtered out later
    
    # Sevastopol variations
    'sevastopol municipality': 'Sevastopol',
    'sevastopol municipality ': 'Sevastopol',
    'sevastopol city state administration': 'Sevastopol',
    
    # Ivano-Frankivsk variations
    'ivanofrankivsk oblast': 'Ivano-Frankivsk Oblast',
    'ivanofrankivsk': 'Ivano-Frankivsk Oblast',
    
    # Odesa/Odessa variations
    'odessa oblast': 'Odesa Oblast',
    'odessa': 'Odesa Oblast',
    
    # Zaporizhzhia variations
    'zaporizhia oblast': 'Zaporizhzhia Oblast',
    'zaporizhia': 'Zaporizhzhia Oblast',
    
    # Crimea variations
    'crimea autonomous republic': 'Crimea',
    'crimea autonomous republic ': 'Crimea',
})

# UPDATED FUNCTION TO HANDLE CLEANING AND FILTERING
def standardize_region(df, col_name='Region'):
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert to string and clean
    df[col_name] = (
        df[col_name]
        .astype(str)
        .str.strip()  # Remove leading/trailing spaces first
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
        .str.replace(r'\d+', '', regex=True)       # Remove digits
        .str.replace(r'\s+', ' ', regex=True)      # Normalize whitespace
        .str.strip()  # Strip again after replacements
    )
    
    # Apply mapping
    df[col_name] = df[col_name].replace(oblast_mapping)
    
    # Remove rows with None values (country-level data)
    df = df[df[col_name].notna()]
    
    # Check for unmapped regions
    unmapped = df[~df[col_name].isin(oblast_mapping.values())][col_name].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Unmapped regions detected: {unmapped}")
    
    return df

# Helper function to clean numeric columns
def clean_numeric(series):
    """Convert series to numeric, removing non-numeric characters"""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^\d\.]', '', regex=True),
        errors='coerce'
    )

# Helper function to standardize names with comprehensive cleaning
def standardize_region(df, col_name='Region'):
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert to string and clean
    df[col_name] = (
        df[col_name]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
        .str.replace(r'\d+', '', regex=True)       # Remove digits
        .str.replace(r'\s+', ' ', regex=True)      # Normalize whitespace
    )
    
    # Apply mapping
    df[col_name] = df[col_name].replace(oblast_mapping)
    
    # Check for unmapped regions
    unmapped = df[~df[col_name].isin(oblast_mapping.values())][col_name].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Unmapped regions detected: {unmapped}")
    
    return df

# Helper function to clean numeric columns
def clean_numeric(series):
    """Convert series to numeric, removing non-numeric characters"""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^\d\.]', '', regex=True),
        errors='coerce'
    )



# ======================
# 2. LOAD AND PROCESS DATASETS
# ======================

print("Step 1/6: Loading conflict data...")
conflict_path = r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\conflict_events_clean.csv"
conflict = pd.read_csv(conflict_path)
conflict = standardize_region(conflict, 'Region')
conflict_agg = conflict.groupby(['Region', 'year']).agg(
    conflict_events=('type_of_violence', 'count'),
    fatalities=('best_est', 'sum')
).reset_index()
print(f"  Processed {len(conflict_agg)} conflict aggregates")
print(f"  Regions in conflict data: {conflict_agg['Region'].nunique()}")

print("Step 2/6: Loading trade data...")
trade_path = r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\Merged_Ukraine_Trade_Data.csv"
trade = pd.read_csv(trade_path)
trade = standardize_region(trade, 'Region')
trade = trade[['Year', 'Region', 'Exports (USD)', 'Imports (USD)', 'Balance']]

# Clean trade numeric columns
trade['Exports (USD)'] = clean_numeric(trade['Exports (USD)'])
trade['Imports (USD)'] = clean_numeric(trade['Imports (USD)'])
trade['Balance'] = clean_numeric(trade['Balance'])

print(f"  Processed {len(trade)} trade records")
print(f"  Regions in trade data: {trade['Region'].nunique()}")

print("Step 3/6: Loading NTL data...")
ntl_path = r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\ntl_results_final.csv"
ntl = pd.read_csv(ntl_path)
ntl = standardize_region(ntl, 'ADM1_EN')
ntl_agg = ntl.groupby(['ADM1_EN', 'year'])['ntl'].mean().reset_index()
print(f"  Processed {len(ntl_agg)} NTL aggregates")
print(f"  Regions in NTL data: {ntl_agg['ADM1_EN'].nunique()}")

print("Step 4/6: Processing population data...")
population_folder = r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\Population By Oblast"
population_dfs = []

# Manual year mapping based on filenames
year_map = {
    'Populations By Oblast 2012.csv': 2012,
    'Population by Oblast 2013.csv': 2013,
    'Population By Oblast 2014.csv': 2014,
    'Population By Oblast 2015.csv': 2015,
    'Population By oblast 2016.csv': 2016,
    'Population By Oblast 2017.csv': 2017,
    'Population By Oblast 2018.csv': 2018,
    'Population By Oblast 2019.csv': 2019,
    'Population By Oblast 2020.csv': 2020,
    'Population By Oblast 2021.csv': 2021,
    'Population By Oblast 2022.csv': 2022
}

# Get all CSV files
all_pop_files = glob(os.path.join(population_folder, "*.csv"))
print(f"  Found {len(all_pop_files)} population files")

# Define possible encodings to try
ENCODINGS = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'windows-1250']

for file_path in all_pop_files:
    filename = os.path.basename(file_path)
    year = year_map.get(filename)
    
    if year is None:
        # Fallback to regex extraction
        match = re.search(r'20\d{2}', filename)
        if match:
            year = int(match.group(0))
            print(f"  ⚠️ Used regex to extract year {year} from {filename}")
        else:
            print(f"  ❌ Could not determine year for {filename}. Skipping.")
            continue
    
    df = None
    # Try different encodings
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"  ✅ Successfully read {filename} with {encoding} encoding")
            break
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception as e:
            print(f"  ❌ Error with encoding {encoding} for {filename}: {str(e)}")
    
    if df is None:
        print(f"  ❌ All encodings failed for {filename}. Skipping.")
        continue
    
    try:
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('.', '') for col in df.columns]
        
        # Handle different file structures
        if "2012" in filename:
            # Special handling for 2012 format
            # Rename first column to Region
            if 'Region' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'Region'})
            # Identify value columns (skip Region column)
            value_cols = df.columns[1:]
            # Melt to long format
            df = df.melt(id_vars=['Region'], 
                         value_vars=value_cols, 
                         var_name='Type', 
                         value_name='Value')
            # Pivot without aggregation
            df = df.pivot(index='Region', columns='Type', values='Value').reset_index()
        else:
            # Rename first column to Region
            df = df.rename(columns={df.columns[0]: 'Region'})
        
        # Identify columns by content
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'region' in col_lower or 'oblast' in col_lower:
                column_mapping[col] = 'Region'
            elif 'urban' in col_lower:
                column_mapping[col] = 'Urban'
            elif 'rural' in col_lower:
                column_mapping[col] = 'Rural'
            elif 'population' in col_lower or 'total' in col_lower:
                column_mapping[col] = 'Total'
        
        # Apply column mapping
        if not column_mapping:
            print(f"  ⚠️ No columns mapped for {filename}. Using first columns.")
            column_mapping = {
                df.columns[0]: 'Region',
                df.columns[1]: 'Total',
                df.columns[2]: 'Urban',
                df.columns[3]: 'Rural'
            }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        if 'Region' not in df.columns or 'Total' not in df.columns:
            print(f"  ❌ Missing required columns in {filename}. Found: {df.columns}")
            continue
            
        # Standardize region names
        df = standardize_region(df, 'Region')
        
        # Build population record
        pop_data = {
            'Region': df['Region'],
            'population_total': clean_numeric(df['Total']),
            'year': year
        }
        
        # Add urban/rural if available
        if 'Urban' in df.columns:
            pop_data['population_urban'] = clean_numeric(df['Urban'])
        if 'Rural' in df.columns:
            pop_data['population_rural'] = clean_numeric(df['Rural'])
        
        population_dfs.append(pd.DataFrame(pop_data))
        print(f"  ✅ Processed {filename} for year {year}")
        
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Combine population data
if population_dfs:
    population_annual = pd.concat(population_dfs)
    print(f"  Combined {len(population_annual)} population records")
else:
    print("  ❗ No population data processed. Using fallback.")
    population_annual = pd.DataFrame(columns=['Region', 'year', 'population_total'])

# Create complete population grid
all_years = pd.DataFrame({'year': range(2012, 2023)})
all_regions = pd.DataFrame({'Region': list(set(oblast_mapping.values()))})
population_complete = (
    all_regions.merge(all_years, how='cross')
    .merge(population_annual, on=['Region', 'year'], how='left')
    .sort_values(['Region', 'year'])
)

# Forward fill missing values within each region
population_complete = population_complete.groupby('Region', group_keys=False).apply(
    lambda x: x.ffill().bfill()
).reset_index(drop=True)

# Calculate urbanization percentage
if 'population_urban' in population_complete.columns and 'population_total' in population_complete.columns:
    population_complete['urban_pct'] = (
        population_complete['population_urban'] / population_complete['population_total']
    )
else:
    population_complete['urban_pct'] = np.nan

print("Step 5/6: Loading general statistics...")
general_path = r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\World Indicators general stats form 12-22.csv"
general = pd.read_csv(general_path)

# Select key indicators
selected_indicators = [
    'GDP (current US$)', 
    'GDP growth (annual %)',
    'Inflation, GDP deflator (annual %)',
    'Merchandise trade (% of GDP)',
    'Foreign direct investment, net inflows (BoP, current US$)'
]

# Debug: Show available indicators
available_indicators = general['Series Name'].unique()
print("Available indicators:")
print(available_indicators)
print(f"Selected indicators: {selected_indicators}")

# Reshape from wide to long
general = general[general['Series Name'].isin(selected_indicators)]
general_long = general.melt(
    id_vars=['Country Name', 'Series Name'], 
    var_name='year_str', 
    value_name='value'
)

# Clean and convert values
general_long['value'] = (
    general_long['value']
    .astype(str)
    .str.replace(r'[^\d\.]', '', regex=True)  # Remove non-numeric characters
    .replace('', np.nan)
    .astype(float)
)

# Extract year from column names
general_long['year'] = general_long['year_str'].str.extract(r'(\d{4})').astype(float).astype('Int64')

# Pivot to wide format by indicator
try:
    general_wide = general_long.pivot_table(
        index=['Country Name', 'year'], 
        columns='Series Name', 
        values='value',
        aggfunc='first'  # Use first value when duplicates
    ).reset_index()
except Exception as e:
    print(f"  ❌ Error creating pivot table: {str(e)}")
    # Handle duplicates explicitly
    general_long = general_long.drop_duplicates(subset=['Country Name', 'year', 'Series Name'])
    general_wide = general_long.pivot_table(
        index=['Country Name', 'year'], 
        columns='Series Name', 
        values='value',
        aggfunc='first'
    ).reset_index()

# ======================
# 3. CREATE BASE DATAFRAME WITH ALL REGIONS
# ======================
print("Step 6/6: Merging all datasets...")
years = range(2012, 2023)

# Explicit list of all 27 oblasts
all_oblasts = [
    'Cherkasy Oblast', 'Chernihiv Oblast', 'Chernivtsi Oblast', 
    'Crimea', 'Dnipropetrovsk Oblast', 'Donetsk Oblast', 
    'Ivano-Frankivsk Oblast', 'Kharkiv Oblast', 'Kherson Oblast', 
    'Khmelnytskyi Oblast', 'Kirovohrad Oblast', 'Kyiv', 
    'Kyiv Oblast', 'Luhansk Oblast', 'Lviv Oblast', 
    'Mykolaiv Oblast', 'Odesa Oblast', 'Poltava Oblast', 
    'Rivne Oblast', 'Sevastopol', 'Sumy Oblast', 
    'Ternopil Oblast', 'Vinnytsia Oblast', 'Volyn Oblast', 
    'Zakarpattia Oblast', 'Zaporizhzhia Oblast', 'Zhytomyr Oblast'
]

base = pd.DataFrame(
    [(oblast, year) for oblast in all_oblasts for year in years],
    columns=['oblast', 'year']
)

# ======================
# 4. MERGE ALL DATASETS WITH VALIDATION
# ======================
print("Merging conflict data...")
merged = base.merge(
    conflict_agg, 
    left_on=['oblast', 'year'], 
    right_on=['Region', 'year'], 
    how='left'
).drop('Region', axis=1)

print("Merging trade data...")
merged = merged.merge(
    trade, 
    left_on=['oblast', 'year'], 
    right_on=['Region', 'Year'], 
    how='left'
).drop(['Region', 'Year'], axis=1)

print("Merging NTL data...")
merged = merged.merge(
    ntl_agg, 
    left_on=['oblast', 'year'], 
    right_on=['ADM1_EN', 'year'], 
    how='left'
).drop('ADM1_EN', axis=1)

print("Merging population data...")
merged = merged.merge(
    population_complete, 
    left_on=['oblast', 'year'], 
    right_on=['Region', 'year'], 
    how='left'
).drop('Region', axis=1)

print("Merging country-level statistics...")
merged = merged.merge(
    general_wide, 
    on='year', 
    how='left'
)

# ======================
# 5. CLEAN AND ENHANCE
# ======================
# Fill missing conflict data with 0
merged[['conflict_events', 'fatalities']] = (
    merged[['conflict_events', 'fatalities']].fillna(0)
)

# Ensure numeric types for calculations
numeric_cols = [
    'Exports (USD)', 'Imports (USD)', 'Balance', 
    'GDP (current US$)', 'ntl', 'population_total'
]

for col in numeric_cols:
    if col in merged.columns:
        merged[col] = clean_numeric(merged[col])

# Create derived metrics
merged['conflict_intensity'] = np.log1p(merged['conflict_events'] + merged['fatalities'])
if 'GDP (current US$)' in merged.columns:
    merged['export_intensity'] = merged['Exports (USD)'] / merged['GDP (current US$)']
    merged['import_intensity'] = merged['Imports (USD)'] / merged['GDP (current US$)']
else:
    merged['export_intensity'] = np.nan
    merged['import_intensity'] = np.nan
    
if 'population_total' in merged.columns and 'ntl' in merged.columns:
    merged['ntl_per_capita'] = merged['ntl'] / merged['population_total']
else:
    merged['ntl_per_capita'] = np.nan

# Handle Crimea special case (no data after 2014)
if 'Crimea' in merged['oblast'].values:
    crimea_mask = (merged['oblast'] == 'Crimea') & (merged['year'] > 2014)
    for col in ['conflict_events', 'fatalities', 'Exports (USD)', 'Imports (USD)', 'Balance']:
        merged.loc[crimea_mask, col] = np.nan

# Final column selection and renaming
final_columns = [
    'year', 'oblast', 
    'population_total', 'population_urban', 'population_rural', 'urban_pct',
    'ntl', 'ntl_per_capita',
    'conflict_events', 'fatalities', 'conflict_intensity',
    'Exports (USD)', 'Imports (USD)', 'Balance',
    'export_intensity', 'import_intensity'
]

# Add available country-level stats
country_stats = [
    'GDP (current US$)', 'GDP growth (annual %)',
    'Inflation, GDP deflator (annual %)',
    'Merchandise trade (% of GDP)',
    'Foreign direct investment, net inflows (BoP, current US$)'
]

for col in country_stats:
    if col in merged.columns:
        final_columns.append(col)

final = merged[final_columns]

# Rename columns
final.columns = [
    'year', 'oblast', 
    'population_total', 'population_urban', 'population_rural', 'urbanization_pct',
    'avg_ntl', 'ntl_per_capita',
    'conflict_events', 'fatalities', 'conflict_intensity',
    'exports_usd', 'imports_usd', 'trade_balance',
    'export_intensity', 'import_intensity'
] + [  # Add country stats with clean names
    'gdp_usd' if col == 'GDP (current US$)' else
    'gdp_growth_pct' if col == 'GDP growth (annual %)' else
    'inflation_pct' if col == 'Inflation, GDP deflator (annual %)' else
    'trade_pct_gdp' if col == 'Merchandise trade (% of GDP)' else
    'fdi_inflows_usd' if col == 'Foreign direct investment, net inflows (BoP, current US$)' else col
    for col in final_columns if col in country_stats
]

# Final cleanup
final = final.dropna(subset=['population_total'], how='all')  # Remove empty rows

# ======================
# 6. VALIDATE AND SAVE
# ======================
# Validation checks
print("\n" + "="*50)
print("VALIDATION CHECKS:")
print(f"Total rows: {len(final)}")
print(f"Expected rows: {27 * 11} = {297} (27 regions × 11 years)")
print(f"Actual regions: {final['oblast'].nunique()}")
print(f"Years: {final['year'].min()} to {final['year'].max()} ({final['year'].nunique()} years)")

# Identify missing regions
missing_regions = set(all_oblasts) - set(final['oblast'].unique())
if missing_regions:
    print(f"❌ Missing regions: {missing_regions}")
else:
    print("✅ All 27 regions present")

# Check for missing population data
pop_missing = final[final['population_total'].isna()]
if not pop_missing.empty:
    print(f"⚠️ Missing population data for {len(pop_missing)} rows")
    print("Affected regions and years:")
    print(pop_missing[['oblast', 'year']].drop_duplicates())
else:
    print("✅ No missing population data")

# Save final dataset
output_path = r"C:\Users\Filk\Desktop\ukraine_research_dataset.csv"
final.to_csv(output_path, index=False)

print("\n" + "="*50)
print(f"✅ Successfully created research dataset with {len(final)} rows")
print(f"📊 Saved to: {output_path}")
print("="*50)

# Show summary
print("\nDataset Summary:")
print(f"- Time range: {final['year'].min()} to {final['year'].max()}")
print(f"- Oblasts: {len(final['oblast'].unique())}")
print(f"- Variables: {len(final.columns)}")
print("\nFirst 3 rows:")
print(final.head(3))