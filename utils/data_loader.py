import pandas as pd
import csv

def load_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
    return pd.read_csv(file_path, sep=dialect.delimiter)