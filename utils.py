import pandas as pd
import zipfile as zf
import os
from settings import PATHS

zip_files = [f for f in os.listdir(PATHS['DATA']) if f.endswith('.zip')]

df_objs = list()

for i in zip_files:
    zip_obj = zf.ZipFile(file=PATHS['DATA'] + i)
    filename = zip_obj.namelist()[0]
    df = pd.read_csv(filepath_or_buffer=zip_obj.open(filename), index_col='id')
    df_objs.append(df)


print(df_objs)
