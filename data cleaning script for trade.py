import pandas as pd
import os
import re
import email
from io import StringIO

def extract_html_from_mhtml(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f)
    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            html = part.get_payload(decode=True)
            if html is None:
                html = part.get_payload()
            if isinstance(html, bytes):
                html = html.decode('utf-8', errors='ignore')
            return html
    raise ValueError("No text/html part found in MHTML file.")

def read_excel_or_mhtml(filepath):
    if filepath.endswith('.mhtml'):
        html_content = extract_html_from_mhtml(filepath)
        tables = pd.read_html(StringIO(html_content))
        print(f"{filepath} contains {len(tables)} tables.")
        for i, table in enumerate(tables):
            print(f"Table {i} shape: {table.shape}")
            print(table.head(3))
        largest_table = max(tables, key=lambda x: x.shape[0]*x.shape[1])
        return largest_table
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return None

# Dictionary to clean/standardize region names to English
region_corrections = {
    'київ': 'kyiv',
    'kyiv city': 'kyiv',
    'м. київ': 'kyiv',
    'київська': 'kyiv oblast',

    'вінницька': 'vinnytsia',
    'vinnytska': 'vinnytsia',

    'волинська': 'volyn',
    'volynska': 'volyn',

    'дніпропетровська': 'dnipropetrovsk',
    'dnipropetrovska': 'dnipropetrovsk',

    'донецька': 'donetsk',
    'донецьк': 'donetsk',
    'donetska': 'donetsk',

    'житомирська': 'zhytomyr',
    'zhytomyrska': 'zhytomyr',

    'закарпатська': 'zakarpattia',
    'transcarpathian': 'zakarpattia',

    'запорізька': 'zaporizhzhia',
    'zaporizka': 'zaporizhzhia',

    'івано-франківська': 'ivano-frankivsk',
    'ivano-frankivska': 'ivano-frankivsk',

    'кіровоградська': 'kirovohrad',
    'kirovohradska': 'kirovohrad',

    'луганська': 'luhansk',
    'луганськ': 'luhansk',
    'luhanska': 'luhansk',

    'львівська': 'lviv',
    'lvivska': 'lviv',

    'миколаївська': 'mykolaiv',
    'mykolaivska': 'mykolaiv',

    'одеська': 'odesa',
    'odeska': 'odesa',

    'полтавська': 'poltava',
    'poltavska': 'poltava',

    'рівненська': 'rivne',
    'rivnenska': 'rivne',

    'сумська': 'sumy',
    'sumskaya': 'sumy',

    'тернопільська': 'ternopil',
    'ternopilska': 'ternopil',

    'харківська': 'kharkiv',
    'kharkivska': 'kharkiv',

    'херсонська': 'kherson',
    'khersonska': 'kherson',

    'хмельницька': 'khmelnytskyi',
    'khmelnytska': 'khmelnytskyi',

    'черкаська': 'cherkasy',
    'cherkaska': 'cherkasy',

    'чернівецька': 'chernivtsi',
    'chernivetska': 'chernivtsi',

    'чернігівська': 'chernihiv',
    'chernihivska': 'chernihiv',

    'севастополь': 'sevastopol',
    'м. севастополь': 'sevastopol',

    'автономна республіка крим': 'crimea',
    'ар крим': 'crimea',
    'crimea republic': 'crimea'
}

def translate_region(region_name):
    if isinstance(region_name, str):
        key = region_name.lower().strip()
        return region_corrections.get(key, region_name)
    return region_name

def read_excel_or_mhtml(filepath):
    if filepath.endswith('.mhtml'):
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        tables = pd.read_html(html_content)
        print(f"{filepath} contains {len(tables)} tables.")
        for i, table in enumerate(tables):
            print(f"Table {i} shape: {table.shape}")
            print(table.head(3))
        # Return the largest table assuming it's the main data
        largest_table = max(tables, key=lambda x: x.shape[0]*x.shape[1])
        return largest_table
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return None

def clean_and_extract_data(df, year):
    print(f"Original df shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    region_col_candidates = [col for col in df.columns if 'region' in str(col).lower() or 'область' in str(col).lower()]
    region_col = region_col_candidates[0] if region_col_candidates else df.columns[0]
    print(f"Selected region column: {region_col}")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    print(f"Numeric columns: {numeric_cols}")

    if len(numeric_cols) >= 2:
        import_col, export_col = numeric_cols[:2]
    else:
        import_col = df.columns[1] if len(df.columns) > 1 else None
        export_col = df.columns[2] if len(df.columns) > 2 else None
    print(f"Import col: {import_col}, Export col: {export_col}")

    df = df.rename(columns={region_col: 'Region'})

    cols = ['Region']
    if import_col is not None:
        cols.append(import_col)
    if export_col is not None:
        cols.append(export_col)
    df = df[cols].copy()
    df['Region'] = df['Region'].apply(translate_region)
    df = df.dropna(subset=['Region'])

    print(f"Data rows after dropping NaN Region: {len(df)}")

    df['Year'] = year

    if import_col is not None:
        df = df.rename(columns={import_col: f'Imports {year}'})
    if export_col is not None:
        df = df.rename(columns={export_col: f'Exports {year}'})

    return df

def process_all_files(folder_path):
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(('.xls', '.xlsx', '.mhtml')):
            years = re.findall(r'20\d{2}', file)
            year = None
            for y in years:
                y_int = int(y)
                if 2012 <= y_int <= 2022:
                    year = y_int
                    break
            if year is None:
                print(f"Skipping {file}: no valid year found.")
                continue

            filepath = os.path.join(folder_path, file)
            print(f"Processing {file} for year {year}...")
            df = read_excel_or_mhtml(filepath)
            if df is not None:
                print(f"Loaded df shape: {df.shape}")
                cleaned_df = clean_and_extract_data(df, year)
                print(f"Cleaned df shape: {cleaned_df.shape}")
                if not cleaned_df.empty:
                    all_data.append(cleaned_df)
                else:
                    print(f"Cleaned dataframe empty for {file}")
            else:
                print(f"Failed to read data from {file}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        cols = ['Year', 'Region'] + [c for c in combined_df.columns if c not in ['Year', 'Region']]
        combined_df = combined_df[cols]
        combined_df = combined_df.sort_values(['Year', 'Region']).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()

# Path to your data folder
folder = r"C:\Users\Filk\Desktop\Personal Research\Trade Dataset"

final_df = process_all_files(folder)
if not final_df.empty:
    final_df.to_csv('cleaned_trade_data.csv', index=False)
    print('CSV file created with', final_df.shape[0], 'rows')
else:
    print('No data processed.')




































