import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

# --- Added helper function to fix misencoded UTF-8 strings ---
def decode_utf8_misinterpreted(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except Exception:
        return text

def parse_html_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        raw_html = f.read()

    # Fix bad rowspan and colspan encodings like '3D"2"' -> '"2"'
    fixed_html = re.sub(r'rowspan=3D"?(\d+)"?', r'rowspan="\1"', raw_html)
    fixed_html = re.sub(r'colspan=3D"?(\d+)"?', r'colspan="\1"', fixed_html)

    soup = BeautifulSoup(fixed_html, 'html.parser')

    # Parse tables using StringIO to avoid FutureWarning
    tables = pd.read_html(StringIO(fixed_html))

    print(f"Found {len(tables)} tables in {os.path.basename(filepath)}")

    # Select largest table (by rows * columns)
    df = max(tables, key=lambda x: x.shape[0] * x.shape[1])

    # Flatten multi-index columns if needed
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(-1)

    # Check if first row is header row with 'region' text and reset columns accordingly
    if df.shape[0] > 0 and df.iloc[0].dtype == object and any(df.iloc[0].str.contains("region", case=False, na=False)):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    # Rename first column to 'region' if it's not named already
    if df.columns[0].lower() not in ['region', 'oblast']:
        df.rename(columns={df.columns[0]: 'region'}, inplace=True)

    # --- Apply UTF-8 fix to region column ---
    df['region'] = df['region'].astype(str).apply(decode_utf8_misinterpreted)

    # Convert all other columns to numeric, cleaning common formatting issues
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace(',', '.')     # replace commas with dots if decimal comma used
            .str.replace('–', '-')     # fix different dash characters
            .str.replace(' ', '')      # remove spaces
            , errors='coerce')

    return df

def read_excel_or_mhtml(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.xls', '.xlsx']:
        return pd.read_excel(filepath)
    elif ext in ['.mhtml', '.html', '.htm']:
        return parse_html_file(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def process_all_files(folder):
    all_dfs = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            print(f"Processing file: {filepath}")
            try:
                df = read_excel_or_mhtml(filepath)
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    # Concatenate all dataframes (assuming they have the same columns)
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df

if __name__ == "__main__":
    folder = r"C:\Users\Filk\Desktop\Personal Research\Trade Dataset"
    final_df = process_all_files(folder)

    print("Final combined dataframe shape:", final_df.shape)

    # Save to CSV file for easy opening in Excel or other programs
    final_df.to_csv(r"C:\Users\Filk\Desktop\Personal Research\cleaned_trade_data.csv", index=False)

    print("Data saved to cleaned_trade_data.csv")




