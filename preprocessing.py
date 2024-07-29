import os
import requests
from zipfile import ZipFile
import subprocess
import platform

# URL and paths
url = 'https://link_to_dataset.com/path_to_jester_dataset.zip'
output_zip = 'jester_dataset.zip'
extract_path = 'path_to_extract'

# Download the dataset
response = requests.get(url, stream=True)
with open(output_zip, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file.write(chunk)

# Extract the dataset
with ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Clean up
os.remove(output_zip)

print("Dataset downloaded and extracted successfully.")

# Open the extracted folder in the file explorer
if platform.system() == 'Windows':
    os.startfile(extract_path)
elif platform.system() == 'Darwin':  # macOS
    subprocess.call(['open', extract_path])
else:  # Linux
    subprocess.call(['xdg-open', extract_path])
