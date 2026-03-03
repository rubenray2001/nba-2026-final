import os
import time
from datetime import datetime

path = "models/betting_metadata.json"
if os.path.exists(path):
    mtime = os.path.getmtime(path)
    print(f"File: {path}")
    print(f"Modified: {datetime.fromtimestamp(mtime)}")
    print(f"Size: {os.path.getsize(path)} bytes")
else:
    print(f"File not found: {path}")
