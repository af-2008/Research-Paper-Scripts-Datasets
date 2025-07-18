import pandas as pd
import requests
import os
from getpass import getpass

# 🔧 SETTINGS
csv_file = r"C:\Users\Filk\Desktop\Perosnal Research\NTLdata.csv"
df = pd.read_csv(csv_file)
output_dir = 'NTL_HDF5_Files'
url_column_name = 'URL'     # change if your CSV column is named differently

# 🔐 Earthdata Login
username = input("af_08 ")
password = getpass("kerosina7R!2008 ")

# 📂 Create output directory
os.makedirs(output_dir, exist_ok=True)

# 📥 Load CSV & extract URLs
df = pd.read_csv(csv_file)
urls = df[url_column_name].dropna().tolist()

# 🔄 Download loop
for url in urls:
    filename = os.path.join(output_dir, os.path.basename(url))
    if os.path.exists(filename):
        print(f"✅ Already downloaded: {filename}")
        continue
    print(f"⬇️ Downloading: {filename}")
    response = requests.get(url, auth=(username, password), stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"❌ Failed to download {filename} (status code {response.status_code})")
