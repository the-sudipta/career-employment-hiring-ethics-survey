import pandas as pd

SHEET_ID = "1cDQ004uMDmlTk6wgvVaD9yHa5a7GxMMW13W6i1GNP8o"
GID = "1834307442"

# Option 1 (most common)
url1 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Option 2 (backup, often works when export fails)
url2 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

try:
    df = pd.read_csv(url1)
except Exception:
    df = pd.read_csv(url2)

print(df.shape)
print(df.head(3))
