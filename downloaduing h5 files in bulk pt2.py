import os
import requests
import csv
from tqdm import tqdm

# === USER CONFIGURATION ===
CSV_PATH = r"C:\Users\Filk\Desktop\Personal Research\Different Data\NTLdata.csv"     # The CSV file with full download URLs
DEST_DIR = r"C:\Users\Filk\Desktop\Personal Research\NTLv4"             # Folder where files will be saved

# === COOKIES FROM BROWSER ===
COOKIES = {
    "ladsweb-auth-session": ".eJxNzkGKwzAMheG7eF2KLMmynVVvEmxZZkJTXJJmMZTefcLMZraP_8H3dnPfbP9y02s77OLmpbnJcVbKUAOw9CyWgKQHjFI8oSIrhk7JQDwTW5CYkANUzbH5BqiCkUA6h6jFkqp5I9aqGFWrx9iQSEo3xpSSVgHBWjDXytzIsy_uhBy7bX-a0mdI5_TcRl9Wc9Pb2aMs61xaO-37b7Ks3_fbOF7rGPerjsfZH__en88PRjRE-g.aG1xSw.VOuzKqbRhZl2ZkbuHtoOan83-NM"
}

# === SETUP ===
os.makedirs(DEST_DIR, exist_ok=True)
session = requests.Session()
session.cookies.update(COOKIES)
session.headers.update({
    'User-Agent': 'python-requests/2.31.0',
    'Accept': '*/*',
})

# === READ CSV AND DOWNLOAD ===
with open(CSV_PATH, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if not row:
            continue
        url = row[0].strip()
        fname = os.path.basename(url)
        dest_path = os.path.join(DEST_DIR, fname)

        if os.path.exists(dest_path):
            print(f"[✓] Already downloaded: {fname}")
            continue

        print(f"[↓] Downloading: {fname} ...")
        try:
            with session.get(url, stream=True, allow_redirects=True) as r:
                if r.status_code == 200 and 'html' not in r.headers.get('Content-Type', ''):
                    with open(dest_path, 'wb') as f:
                        for chunk in tqdm(r.iter_content(chunk_size=8192)):
                            if chunk:
                                f.write(chunk)
                    print(f"[✓] Done: {fname}")
                else:
                    print(f"[✗] Failed: {fname} (Status {r.status_code}, Content-Type: {r.headers.get('Content-Type', '')})")
        except Exception as e:
            print(f"[ERROR] {fname} - {e}")

