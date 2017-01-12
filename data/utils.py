import pandas as pd
import zipfile as zf
from settings import PATHS

zip_obj = zf.ZipFile(file=PATHS['ZIP'] + 'travel.csv.zip')
sample = pd.read_csv(filepath_or_buffer=zip_obj.open('travel.csv'))

print(sample.head(20))
