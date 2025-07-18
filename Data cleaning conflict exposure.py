import pandas as pd
import os
from dateutil.parser import parse

# Flexible date parser to handle mixed formats
def parse_flexibly(date_str):
    try:
        return parse(str(date_str), dayfirst=False).strftime('%Y-%m-%d')
    except Exception:
        return "INVALID_DATE"

# Cleaning function
def clean_conflict_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path, encoding='utf-8')

    # Normalize and parse both date columns
    for col in ['date_start', 'date_end']:
        df[col] = df[col].astype(str).str.strip()                 # Clean whitespace
        df[col] = df[col].apply(parse_flexibly)                  # Parse to consistent format

    # Extract year from cleaned date_start
    df['year'] = df['date_start'].str[:4]

    # Select only relevant columns
    keep_cols = [
        'year',
        'type_of_violence',
        'adm_1',
        'best_est',
        'latitude',
        'longitude',
        'date_start',
        'date_end',
        'event_clarity'
    ]
    df = df[[col for col in keep_cols if col in df.columns]]

    # Save to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Cleaned dataset saved to: {output_path}")

# Run the cleaning script
if __name__ == "__main__":
    input_file = r"C:\Users\Filk\Desktop\Personal Research\Conflict Data\conflict_events_raw.csv"
    output_file = r"C:\Users\Filk\Desktop\Personal Research\Clean data\conflict_events_clean.csv"

    clean_conflict_data(input_file, output_file)




