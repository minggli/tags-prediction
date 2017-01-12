import pandas as pd
import os
import zipfile as zf
from settings import PATHS


def zip_to_df(path):

    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    df_objs = list()

    for i in zip_files:
        zip_obj = zf.ZipFile(file=path + i)
        filename = zip_obj.namelist()[0]
        df = pd.read_csv(filepath_or_buffer=zip_obj.open(filename), index_col='id')
        zip_obj.close()
        df_objs.append(df)
        print(filename)

    return df_objs
